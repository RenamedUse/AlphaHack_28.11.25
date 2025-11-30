import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import lightgbm as lgb
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
import joblib
import pickle
import gc
import shap

def weighted_mean_absolute_error(y_true, y_pred, weights):
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

def load_and_preprocess(train_path, test_path):
    train = pd.read_csv(train_path, sep=";", decimal=",", low_memory=False)
    test = pd.read_csv(test_path, sep=";", decimal=",", low_memory=False)
    
    y = train["target"].values.astype(float)
    weights = train["w"].values.astype(float)
    drop_cols = ["dt", "id", "w", "target"]
    
    df = pd.concat([train, test], axis=0, ignore_index=True)
    
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    orig_num_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop(drop_cols, errors='ignore').tolist()
    
    medians = {}
    for col in orig_num_cols:
        medians[col] = df[col].median()
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(medians[col])
    
    maps = {}
    for col in cat_cols:
        df[col] = df[col].fillna("MISSING")
        unique_vals = df[col].unique()
        maps[col] = {v: i for i, v in enumerate(unique_vals)}
        df[col] = df[col].map(maps[col])
    
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
    
    upper_limits = {}
    for col in all_num_cols:
        if col in df.columns:
            upper_limits[col] = df[col].quantile(0.99)
            df[col] = np.clip(df[col], None, upper_limits[col])
    
    turnover_cols = [col for col in df.columns if 'turn_' in col]
    pca = None
    if turnover_cols:
        pca = PCA(n_components=5)
        pca_features = pca.fit_transform(df[turnover_cols])
        pca_df = pd.DataFrame(pca_features, columns=[f'pca_turn_{i}' for i in range(5)])
        df = pd.concat([df, pca_df], axis=1)
        df.drop(columns=turnover_cols, inplace=True)
    
    pca_cols = [f'pca_turn_{i}' for i in range(5)] if turnover_cols else []
    all_num_cols = [col for col in all_num_cols if col not in turnover_cols] + pca_cols
    
    scaler = RobustScaler()
    df[all_num_cols] = scaler.fit_transform(df[all_num_cols])
    
    train_proc = df.iloc[:len(train)].copy()
    test_proc = df.iloc[len(train):].copy()
    
    del df
    gc.collect()
    
    X = train_proc.drop(columns=drop_cols, errors="ignore")
    X_test = test_proc.drop(columns=drop_cols, errors="ignore")
    
    features = X.columns.tolist()
    
    return X, X_test, y, weights, scaler, features, medians, upper_limits, pca, maps

def train_base_models(X, y, weights, features, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    y_log = np.log1p(y)
    
    lgb_oof = np.zeros(len(X))
    cat_oof = np.zeros(len(X))
    nn_oof = np.zeros(len(X))
    
    lgb_models = []
    cat_models = []
    nn_models = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X), 1):
        print(f"Fold {fold}/{n_folds}")
        
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr_log, y_val_log = y_log[train_idx], y_log[valid_idx]
        y_tr, y_val = y[train_idx], y[valid_idx]
        w_tr, w_val = weights[train_idx], weights[valid_idx]
        
        # LightGBM
        lgb_params = {
            'objective': 'mae',
            'metric': 'mae',
            'learning_rate': 0.02,
            'max_depth': 10,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42 + fold,
            'verbose': -1
        }
        lgb_train = lgb.Dataset(X_tr, label=y_tr_log, weight=w_tr)
        lgb_valid = lgb.Dataset(X_val, label=y_val_log, weight=w_val)
        lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=5000,
                              valid_sets=[lgb_train, lgb_valid],
                              callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)])
        lgb_oof[valid_idx] = np.expm1(lgb_model.predict(X_val)) 
        lgb_models.append(lgb_model)
        
        # CatBoost
        cat_params = {
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
            'learning_rate': 0.02,
            'depth': 8,
            'subsample': 0.8,
            'colsample_bylevel': 0.8,
            'l2_leaf_reg': 3,
            'random_seed': 42 + fold,
            'verbose': 500,
            'iterations': 5000,
            'early_stopping_rounds': 200
        }
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(X_tr, y_tr_log, sample_weight=w_tr,
                      eval_set=(X_val, y_val_log), use_best_model=True)
        cat_oof[valid_idx] = np.expm1(cat_model.predict(X_val))
        cat_models.append(cat_model)
        
        # NN Keras
        nn_model = Sequential([
            Dense(128, activation='relu', input_shape=(X_tr.shape[1],)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
        nn_model.fit(X_tr, y_tr_log, sample_weight=w_tr, epochs=50, batch_size=256,
                     validation_data=(X_val, y_val_log, w_val), verbose=0)
        nn_oof[valid_idx] = np.expm1(nn_model.predict(X_val, verbose=0).flatten())
        nn_models.append(nn_model)
        
        gc.collect()
    
    weighted_mean_absolute_error(y, lgb_oof, weights)
    weighted_mean_absolute_error(y, cat_oof, weights)
    weighted_mean_absolute_error(y, nn_oof, weights)
    
    return lgb_models, cat_models, nn_models, np.column_stack([lgb_oof, cat_oof, nn_oof])

def train_meta_model(oof_stack, y, weights):
    meta_model = xgb.XGBRegressor(objective='reg:absoluteerror', learning_rate=0.01, max_depth=6,
                                  subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=1,
                                  random_state=42, tree_method='hist', n_estimators=2000)
    meta_model.fit(oof_stack, np.log1p(y), sample_weight=weights)
    return meta_model

def predict_test(base_models, meta_model, X_test):
    lgb_preds = np.mean([np.expm1(m.predict(X_test)) for m in base_models[0]], axis=0)
    cat_preds = np.mean([np.expm1(m.predict(X_test)) for m in base_models[1]], axis=0)
    nn_preds = np.mean([np.expm1(m.predict(X_test, verbose=0).flatten()) for m in base_models[2]], axis=0)
    stack_preds = np.column_stack([lgb_preds, cat_preds, nn_preds])
    return np.expm1(meta_model.predict(stack_preds))

def create_shap_explainer(meta_model, oof_stack_sample):
    explainer = shap.Explainer(meta_model)
    shap_values = explainer(oof_stack_sample)  
    return explainer

def main(train_path="hackathon_income_train.csv", test_path="hackathon_income_test.csv"):
    X, X_test, y, weights, scaler, features, medians, upper_limits, pca, maps = load_and_preprocess(train_path, test_path)
    
    lgb_models, cat_models, nn_models, oof_stack = train_base_models(X, y, weights, features)
    
    meta_model = train_meta_model(oof_stack, y, weights)
    
    ensemble_oof = np.expm1(meta_model.predict(oof_stack))
    print(f"Ensemble OOF WMAE: {weighted_mean_absolute_error(y, ensemble_oof, weights):.5f}")
    
    test_preds = predict_test((lgb_models, cat_models, nn_models), meta_model, X_test)
    
    submission = pd.DataFrame({"id": pd.read_csv(test_path, sep=";", usecols=["id"])["id"], "target": test_preds})
    submission.to_csv("submission_ensemble.csv", index=False)
    
    joblib.dump({
        'lgb_models': lgb_models,
        'cat_models': cat_models,
        'nn_models': nn_models,
        'meta_model': meta_model,
        'scaler': scaler,
        'features': features,
        'medians': medians,
        'upper_limits': upper_limits,
        'pca': pca,
        'maps': maps
    }, "income_ensemble_model.pkl")
    
    oof_sample = oof_stack[:100]
    explainer = create_shap_explainer(meta_model, oof_sample)
    joblib.dump(explainer, "shap_explainer.pkl")
    
    preprocessor = {"features": features}
    with open("preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    
    pass

if __name__ == "__main__":
    main()