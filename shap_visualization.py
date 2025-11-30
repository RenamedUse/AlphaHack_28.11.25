import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

def load_model_and_data():
    model_bundle = joblib.load("income_ensemble_model.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    test_df = pd.read_csv("hackathon_income_test.csv", sep=";", decimal=",", nrows=5)
    return model_bundle, explainer, test_df

def preprocess_sample(client_data, model_bundle):
    scaler = model_bundle['scaler']
    features = model_bundle['features']
    medians = model_bundle['medians']
    upper_limits = model_bundle['upper_limits']
    pca = model_bundle['pca']
    maps = model_bundle['maps']

    df = pd.DataFrame([client_data])

    orig_num_cols = list(medians.keys())
    for col in orig_num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(medians[col])

    cat_cols = list(maps.keys())
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("MISSING")
            df[col] = df[col].apply(lambda v: maps[col].get(v, maps[col].get("MISSING", -1)))

    new_features = {
        'income_to_age_ratio': df['incomeValue'] / (df['age'] + 1e-5),
        'turnover_ratio_cr_db': df['turn_cur_cr_avg_v2'] / (df['turn_cur_db_avg_v2'] + 1e-5),
        'bki_limit_to_income': df['hdb_bki_total_max_limit'] / (df['incomeValue'] + 1e-5),
        'salary_growth_1y3y': df['dp_ils_salary_ratio_1y3y'],
        'total_turnover': df['turn_cur_cr_sum_v2'] + df['turn_cur_db_sum_v2']
    }
    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)

    new_num_cols = ['income_to_age_ratio', 'turnover_ratio_cr_db', 'bki_limit_to_income', 'salary_growth_1y3y', 'total_turnover']

    all_num_cols = orig_num_cols + new_num_cols

    for col in all_num_cols:
        if col in df.columns and col in upper_limits:
            df[col] = np.clip(df[col], None, upper_limits[col])

    turnover_cols = [col for col in df.columns if 'turn_' in col]
    if pca and turnover_cols:
        pca_features = pca.transform(df[turnover_cols])
        pca_df = pd.DataFrame(pca_features, columns=[f'pca_turn_{i}' for i in range(5)])
        df = pd.concat([df, pca_df], axis=1)
        df.drop(columns=turnover_cols, inplace=True)

    pca_cols = [f'pca_turn_{i}' for i in range(5)] if turnover_cols else []
    all_num_cols = [col for col in all_num_cols if col not in turnover_cols] + pca_cols

    df[all_num_cols] = scaler.transform(df[all_num_cols])

    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]

def predict_with_shap(client_data, model_bundle, explainer):
    lgb_models = model_bundle['lgb_models']
    cat_models = model_bundle['cat_models']
    nn_models = model_bundle['nn_models']
    meta_model = model_bundle['meta_model']

    X_client = preprocess_sample(client_data, model_bundle)

    lgb_preds = np.mean([np.expm1(m.predict(X_client)) for m in lgb_models], axis=0)
    cat_preds = np.mean([np.expm1(m.predict(X_client)) for m in cat_models], axis=0)
    nn_preds = np.mean([np.expm1(m.predict(X_client.values, verbose=0).flatten()) for m in nn_models], axis=0)

    stack_preds = np.column_stack([lgb_preds, cat_preds, nn_preds])
    pred = np.expm1(meta_model.predict(stack_preds))[0]

    shap_values = explainer(stack_preds)

    return pred, shap_values

if __name__ == "__main__":
    model_bundle, explainer, test_df = load_model_and_data()

    client_data = test_df.iloc[0].to_dict()
    client_data.pop('id', None)
    client_data.pop('dt', None)

    pred, shap_values = predict_with_shap(client_data, model_bundle, explainer)

    print(f"Predicted income: {pred:.2f}")

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.title(f"SHAP Explanation for Predicted Income: {pred:.2f}")
    plt.tight_layout()
    plt.savefig("shap_waterfall.png", dpi=150, bbox_inches='tight')
    plt.show()

    all_shap = []
    preds = []
    for i in range(min(5, len(test_df))):
        client_data = test_df.iloc[i].to_dict()
        client_data.pop('id', None)
        client_data.pop('dt', None)
        pred, shap_val = predict_with_shap(client_data, model_bundle, explainer)
        all_shap.append(shap_val.values[0])
        preds.append(pred)

    all_shap = np.array(all_shap)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(all_shap, feature_names=['LGB Prediction', 'CatBoost Prediction', 'NN Prediction'],
                      show=False, max_display=3)
    plt.title("SHAP Summary Plot for Ensemble Meta-Model")
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches='tight')
    plt.show()
