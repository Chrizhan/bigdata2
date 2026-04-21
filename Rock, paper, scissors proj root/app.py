from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import cv2
import base64

from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.measure import label, regionprops

app = Flask(__name__)

# =========================
# LOAD TRAINED FILES
# =========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
indices = joblib.load("selected_features.pkl")
le = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

# =========================
# SAFE STAT (FROM COLAB)
# =========================
def safe_stat(func, arr, default=0.0):
    arr = np.asarray(arr).astype(np.float32).ravel()
    if arr.size == 0:
        return float(default)
    try:
        val = func(arr)
        if np.isnan(val) or np.isinf(val):
            return float(default)
        return float(val)
    except:
        return float(default)

# =========================
# FEATURE FUNCTIONS (EXACT FROM COLAB)
# =========================

def extract_color_features(img_rgb, img_gray, img_hsv, img_lab):
    feats = {}

    # RGB
    for i, name in enumerate(['r','g','b']):
        ch = img_rgb[:,:,i]
        feats[f'rgb_mean_{name}'] = np.mean(ch)
        feats[f'rgb_std_{name}'] = np.std(ch)
        feats[f'rgb_skew_{name}'] = safe_stat(skew, ch)

    # HSV
    for i, name in enumerate(['h','s','v']):
        ch = img_hsv[:,:,i]
        feats[f'hsv_mean_{name}'] = np.mean(ch)
        feats[f'hsv_std_{name}'] = np.std(ch)

    # LAB
    for i, name in enumerate(['l','a','b']):
        ch = img_lab[:,:,i]
        feats[f'lab_mean_{name}'] = np.mean(ch)
        feats[f'lab_std_{name}'] = np.std(ch)

    # GRAY
    feats['gray_mean'] = np.mean(img_gray)
    feats['gray_std'] = np.std(img_gray)
    feats['gray_skew'] = safe_stat(skew, img_gray)
    feats['gray_kurtosis'] = safe_stat(kurtosis, img_gray)

    hist, _ = np.histogram(img_gray, bins=256, range=(0,256), density=True)
    feats['gray_entropy'] = -np.sum(hist * np.log2(hist + 1e-12))

    # RGB HIST (8 bins)
    for i, name in enumerate(['r','g','b']):
        hist = cv2.calcHist([img_rgb],[i],None,[8],[0,256]).flatten()
        hist = hist/(hist.sum()+1e-12)
        for j,val in enumerate(hist):
            feats[f'rgb_hist_{name}_{j}'] = val

    return feats


def extract_glcm_features(img_gray):
    feats = {}

    glcm = graycomatrix(img_gray, [1,2],
                        [0,np.pi/4,np.pi/2,3*np.pi/4],
                        256, symmetric=True, normed=True)

    for prop in ['contrast','correlation','energy','homogeneity']:
        values = graycoprops(glcm, prop).flatten()
        feats[f'glcm_{prop}_mean'] = np.mean(values)
        feats[f'glcm_{prop}_std'] = np.std(values)
        feats[f'glcm_{prop}_skew'] = safe_stat(skew, values)
        feats[f'glcm_{prop}_kurtosis'] = safe_stat(kurtosis, values)

    return feats


def extract_lbp_features(img_gray):
    feats = {}

    lbp = local_binary_pattern(img_gray, 8, 1, method="uniform")
    n_bins = int(lbp.max()+1)

    hist,_ = np.histogram(lbp.ravel(), bins=n_bins, range=(0,n_bins), density=True)

    for i,val in enumerate(hist):
        feats[f'lbp_hist_{i}'] = val

    feats['lbp_mean'] = np.mean(lbp)
    feats['lbp_std'] = np.std(lbp)

    return feats


def create_hand_mask(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # normalize lighting
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # adaptive threshold (better than Otsu for camera)
    mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # stronger cleanup (important for scissors fingers)
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def extract_shape_features(img_rgb):
    feats = {}

    mask = create_hand_mask(img_rgb)

    labeled = label(mask > 0)
    props = regionprops(labeled)

    # If no object detected
    if len(props) == 0:
        keys = [
            'area','perimeter','bbox_w','bbox_h',
            'aspect_ratio','extent','solidity',
            'equiv_diameter','eccentricity',
            'contour_area','contour_perimeter',
            'hull_area','solidity_contour'
        ]
        for k in keys:
            feats[k] = 0.0
        return feats

    # Largest region (hand)
    region = max(props, key=lambda x: x.area)

    minr, minc, maxr, maxc = region.bbox
    h = maxr - minr
    w = maxc - minc

    feats['area'] = float(region.area)
    feats['perimeter'] = float(region.perimeter)
    feats['bbox_w'] = float(w)
    feats['bbox_h'] = float(h)
    feats['aspect_ratio'] = float(w / (h + 1e-12))
    feats['extent'] = float(region.extent)
    feats['solidity'] = float(region.solidity)

    # KEEP THIS (important for consistency)
    feats['equiv_diameter'] = float(region.equivalent_diameter)

    feats['eccentricity'] = float(region.eccentricity)

    feats['finger_separation'] = float(region.perimeter / (region.area + 1e-6))

    feats['circularity'] = float((4 * np.pi * region.area) / ((region.perimeter ** 2) + 1e-6))

    feats['area_ratio'] = float(region.area / (224 * 224))

    # =========================
    # CONTOUR FEATURES (NEW)
    # =========================
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        feats['contour_area'] = 0.0
        feats['contour_perimeter'] = 0.0
        feats['hull_area'] = 0.0
        feats['solidity_contour'] = 0.0
    else:
        cnt = max(contours, key=cv2.contourArea)

        contour_area = cv2.contourArea(cnt)
        contour_perimeter = cv2.arcLength(cnt, True)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        feats['contour_area'] = float(contour_area)
        feats['contour_perimeter'] = float(contour_perimeter)
        feats['hull_area'] = float(hull_area)
        feats['convexity'] = float(feats['contour_perimeter'] / (cv2.arcLength(cv2.convexHull(cnt), True) + 1e-6))

        if hull_area != 0:
            feats['solidity_contour'] = float(contour_area / hull_area)
        else:
            feats['solidity_contour'] = 0.0

    return feats


def extract_hog_features(img_gray):
    feats = {}

    hog_vec = hog(img_gray,
                  orientations=9,
                  pixels_per_cell=(32,32),
                  cells_per_block=(2,2),
                  block_norm="L2-Hys",
                  feature_vector=True)

    for i,val in enumerate(hog_vec):
        feats[f'hog_{i}']=val

    gx = cv2.Sobel(img_gray,cv2.CV_64F,1,0)
    gy = cv2.Sobel(img_gray,cv2.CV_64F,0,1)
    grad = np.sqrt(gx**2 + gy**2)

    feats['grad_mean']=np.mean(grad)
    feats['grad_std']=np.std(grad)

    return feats


def extract_region_features(img_rgb):
    feats={}

    mask = create_hand_mask(img_rgb)
    labeled = label(mask>0)
    props = regionprops(labeled)

    total_area = np.sum(mask>0)

    feats['region_area']=total_area
    feats['region_count']=len(props)
    feats['region_ratio']=total_area/(224*224) if total_area>0 else 0

    if len(props)==0:
        feats['mean_region_area']=0
        feats['largest_region_area']=0
        feats['region_perimeter_sum']=0
    else:
        areas=[p.area for p in props]
        per=[p.perimeter for p in props]

        feats['mean_region_area']=np.mean(areas)
        feats['largest_region_area']=np.max(areas)
        feats['region_perimeter_sum']=np.sum(per)

    return feats


# =========================
# FINAL FEATURE EXTRACTION
# =========================
def extract_features(image):
    image = cv2.resize(image, (224, 224))

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    features = {}

    features.update(extract_color_features(img_rgb, img_gray, img_hsv, img_lab))
    features.update(extract_glcm_features(img_gray))
    features.update(extract_lbp_features(img_gray))
    features.update(extract_shape_features(img_rgb))
    features.update(extract_hog_features(img_gray))
    features.update(extract_region_features(img_rgb))


    feature_vector = pd.DataFrame([features])
    feature_vector = feature_vector[feature_names]

    return feature_vector


# =========================
# PROCESS IMAGE
# =========================
def process_image(image):
    features = extract_features(image)

    print("BEFORE SCALING:", features.iloc[0, :10].values)

    features = imputer.transform(features)
    features = scaler.transform(features)

    print("AFTER SCALING:", features[0][:10])

    features = features[:, indices]

    pred = model.predict(features)
    return le.inverse_transform(pred)[0]


# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    result = process_image(img)
    return render_template('index.html', prediction=result)


@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    data = request.json['image']
    encoded = data.split(',')[1]
    img_bytes = base64.b64decode(encoded)

    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    result = process_image(img)
    return jsonify({'prediction': result})

# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)