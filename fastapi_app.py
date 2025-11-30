from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional, Union
from contextlib import asynccontextmanager

model_bundle = None
explainer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_bundle, explainer
    model_bundle = joblib.load("income_ensemble_model.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    yield

app = FastAPI(title="Income Prediction API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ClientData(BaseModel):
    id: Optional[int] = None
    dt: Optional[str] = None
    turn_cur_cr_avg_act_v2: Optional[float] = None
    salary_6to12m_avg: Optional[float] = None
    hdb_bki_total_max_limit: Optional[float] = None
    dp_ils_paymentssum_avg_12m: Optional[float] = None
    hdb_bki_total_cc_max_limit: Optional[float] = None
    incomeValue: Optional[float] = None
    gender: Optional[str] = None
    avg_cur_cr_turn: Optional[float] = None
    adminarea: Optional[str] = None
    turn_cur_cr_avg_v2: Optional[float] = None
    turn_cur_cr_max_v2: Optional[float] = None
    hdb_bki_total_pil_max_limit: Optional[float] = None
    age: Optional[float] = None
    dp_ils_avg_salary_1y: Optional[float] = None
    turn_cur_cr_sum_v2: Optional[float] = None
    by_category__amount__sum__eoperation_type_name__ishodjaschij_bystryj_platezh_sbp: Optional[float] = None
    turn_cur_db_sum_v2: Optional[float] = None
    turn_cur_db_avg_act_v2: Optional[float] = None
    dp_ils_avg_salary_2y: Optional[float] = None
    curr_rur_amt_cm_avg: Optional[float] = None
    turn_cur_db_avg_v2: Optional[float] = None
    by_category__amount__sum__eoperation_type_name__vhodjaschij_bystryj_platezh_sbp: Optional[float] = None
    dp_ils_paymentssum_avg_6m: Optional[float] = None
    avg_cur_db_turn: Optional[float] = None
    hdb_bki_active_cc_max_limit: Optional[float] = None
    incomeValueCategory: Optional[float] = None
    avg_by_category__amount__sum__cashflowcategory_name__vydacha_nalichnyh_v_bankomate: Optional[float] = None
    avg_credit_turn_rur: Optional[float] = None
    dp_ils_salary_ratio_1y3y: Optional[float] = None
    by_category__amount__sum__eoperation_type_name__perevod_po_nomeru_telefona: Optional[float] = None
    turn_cur_cr_7avg_avg_v2: Optional[float] = None
    dp_ils_accpayment_avg_12m: Optional[float] = None
    curbal_usd_amt_cm_avg: Optional[float] = None
    avg_by_category__amount__sum__cashflowcategory_name__supermarkety: Optional[float] = None
    avg_loan_cnt_with_insurance: Optional[float] = None
    avg_by_category__amount__sum__cashflowcategory_name__gipermarkety: Optional[float] = None
    city_smart_name: Optional[str] = None
    uniV5: Optional[float] = None
    turn_cur_db_max_v2: Optional[float] = None
    avg_by_category__amount__sum__cashflowcategory_name__kafe: Optional[float] = None
    turn_other_db_max_v2: Optional[float] = None
    turn_cur_cr_min_v2: Optional[float] = None
    hdb_bki_other_active_pil_outstanding: Optional[float] = None
    dp_ewb_last_employment_position: Optional[str] = None
    turn_cur_db_min_v2: Optional[float] = None
    hdb_bki_total_products: Optional[float] = None
    per_capita_income_rur_amt: Optional[float] = None
    avg_debet_turn_rur: Optional[float] = None
    hdb_relend_active_max_psk: Optional[float] = None
    dda_rur_amt_curr_v2: Optional[float] = None
    mob_cnt_days: Optional[float] = None
    dp_ils_days_from_last_doc: Optional[float] = None
    avg_6m_money_transactions: Optional[float] = None
    transaction_category_supermarket_percent_cnt_2m: Optional[float] = None
    pil: Optional[float] = None
    hdb_bki_total_max_overdue_sum: Optional[float] = None
    avg_6m_clothing: Optional[float] = None
    avg_by_category__amount__sum__cashflowcategory_name__elektronnye_dengi: Optional[float] = None
    addrref: Optional[str] = None
    bki_total_auto_cnt: Optional[float] = None
    dp_payoutincomedata_payout_avg_3_month: Optional[float] = None
    hdb_outstand_sum: Optional[float] = None
    avg_3m_money_transactions: Optional[float] = None
    dp_address_unique_regions: Optional[float] = None
    min_balance_rur_amt_6m_af: Optional[float] = None
    transaction_category_supermarket_sum_cnt_m3_4: Optional[float] = None
    dp_payoutincomedata_payout_max_3_month: Optional[float] = None
    hdb_bki_total_ip_max_limit: Optional[float] = None
    hdb_bki_total_cnt: Optional[float] = None
    blacklist_flag: Optional[float] = None
    bki_total_oth_cnt: Optional[float] = None
    dp_payoutincomedata_payout_sum_3_month: Optional[float] = None
    hdb_relend_outstand_sum: Optional[float] = None
    total_rur_amt_cm_avg: Optional[float] = None
    mob_cover_days: Optional[float] = None
    dp_payoutincomedata_payout_max_6_month: Optional[float] = None
    label_Below_50k_share_r1: Optional[float] = None
    turn_fdep_db_sum_v2: Optional[float] = None
    dp_ils_accpayment_avg_6m_current: Optional[float] = None
    transaction_category_cash_percent_amt_2m: Optional[float] = None
    curr_rur_amt_3m_avg: Optional[float] = None
    transaction_category_restaurants_sum_amt_m2: Optional[float] = None
    loan_cnt: Optional[float] = None
    turn_fdep_db_avg_v2: Optional[float] = None
    turn_cur_db_7avg_avg_v2: Optional[float] = None
    bki_total_ip_max_outstand: Optional[float] = None
    amount_by_category_90d__summarur_amt__sum__cashflowcategory_name__vydacha_nalichnyh_v_bankomate: Optional[float] = None
    profit_income_out_rur_amt_12m: Optional[float] = None
    avg_6m_hotels: Optional[float] = None
    hdb_ovrd_sum: Optional[float] = None
    dp_ils_total_seniority: Optional[float] = None
    dp_ils_paymentssum_avg_6m_current: Optional[float] = None
    smsInWavg6m: Optional[float] = None
    avg_fdep_db_turn: Optional[float] = None
    device_iphone_avg: Optional[float] = None
    by_category__amount__sum__eoperation_type_name__platezh_za_mobilnyj_cherez_ps: Optional[float] = None
    avg_balance_rur_amt_1m_af: Optional[float] = None
    curr_rur_amt_cm_avg_period_days_ago_v2: Optional[float] = None
    avg_by_category__amount__sum__cashflowcategory_name__oteli: Optional[float] = None
    hdb_bki_total_ip_cnt: Optional[float] = None
    hdb_bki_active_cc_max_outstand: Optional[float] = None
    hdb_other_outstand_sum: Optional[float] = None
    days_to_last_transaction: Optional[float] = None
    hdb_bki_total_pil_max_overdue: Optional[float] = None
    vert_pil_last_credit_step_screen_view_3m: Optional[float] = None
    acard: Optional[float] = None
    bki_total_il_max_limit: Optional[float] = None
    other_credits_count: Optional[float] = None
    tz_msk_timedelta: Optional[float] = None
    turn_save_db_min_v2: Optional[float] = None
    profit_income_out_rur_amt_9m: Optional[float] = None
    dp_ils_ipkcurrentyear_currentyearpensfactor: Optional[float] = None
    avg_by_category__amount__sum__cashflowcategory_name__odezhda: Optional[float] = None
    cntOnnRinCallAvg6m: Optional[float] = None
    dda_rur_amt_3m_avg: Optional[float] = None
    winback_cnt: Optional[float] = None
    salary_median_in_gex_r1: Optional[float] = None
    dp_payoutincomedata_payout_avg_prev_year: Optional[float] = None
    avg_amount_daily_transactions_90d: Optional[float] = None
    vert_has_app_ru_tinkoff_investing: Optional[float] = None
    transaction_category_supermarket_inc_cnt_2m: Optional[float] = None
    vert_pil_sms_success_3m: Optional[float] = None
    min_balance_rur_amt_1m_af: Optional[float] = None
    dp_ils_max_seniority: Optional[float] = None
    avg_by_category__amount__sum__cashflowcategory_name__set_supermarketov: Optional[float] = None
    label_500k_to_1M_share_r1: Optional[float] = None
    avg_by_category__amount__sum__cashflowcategory_name__zarubezhnye_finansovye_operatsii: Optional[float] = None
    bki_total_products: Optional[float] = None
    avg_6m_all: Optional[float] = None
    dp_ils_avg_simultanious_jobs_5y: Optional[float] = None
    dp_ewb_dismissal_due_contract_violation_by_lb_cnt: Optional[float] = None
    summarur_1m_purch: Optional[float] = None
    diff_avg_cr_db_turn: Optional[float] = None
    dp_ils_cnt_changes_1y: Optional[float] = None
    dp_ils_employeers_cnt_last_month: Optional[float] = None
    dp_payoutincomedata_payout_avg_6_month: Optional[float] = None
    dp_ewb_last_organization: Optional[Union[str, int]] = None
    by_category__amount__sum__eoperation_type_name__perevod_mezhdu_svoimi_schetami: Optional[float] = None
    bki_active_auto_cnt: Optional[float] = None
    turn_other_cr_avg_act_v2: Optional[float] = None
    cntVoiceOutMob6m: Optional[float] = None
    avg_by_category__amount__sum__cashflowcategory_name__puteshestvija: Optional[float] = None
    loanacc_rur_amt_cm_avg: Optional[float] = None
    transaction_category_supermarket_sum_cnt_m2: Optional[float] = None
    transaction_category_supermarket_sum_amt_d15: Optional[float] = None
    avg_fdep_cr_turn: Optional[float] = None
    transaction_category_restaurants_percent_cnt_2m: Optional[float] = None
    bki_total_max_limit: Optional[float] = None
    avg_by_category__amount__sum__cashflowcategory_name__reklama_v_internete: Optional[float] = None
    transaction_category_restaurants_percent_amt_2m: Optional[float] = None
    turn_fdep_db_avg_act_v2: Optional[float] = None
    dp_ils_accpayment_avg_6m: Optional[float] = None
    turn_other_cr_sum_v2: Optional[float] = None
    client_active_flag: Optional[float] = None
    avg_by_category__amount__sum__cashflowcategory_name__produkty: Optional[float] = None
    curr_rur_amt_cm_avg_inc_v2: Optional[float] = None
    nonresident_flag: Optional[float] = None
    avg_by_category__amount__sum__cashflowcategory_name__kosmetika: Optional[float] = None
    vert_has_app_ru_vtb_invest: Optional[float] = None
    dp_ils_avg_salary_3y: Optional[float] = None
    hdb_bki_total_auto_max_limit: Optional[float] = None
    days_after_last_request: Optional[float] = None
    cntRegionTripsWavg1m: Optional[float] = None
    vert_has_app_ru_cian_main: Optional[float] = None
    loanacc_rur_amt_curr_v2: Optional[float] = None
    avg_3m_no_cat: Optional[float] = None
    vert_ghost_close_dpay3_last_days: Optional[float] = None
    vert_has_app_ru_raiffeisennews: Optional[float] = None
    dp_ils_days_ip_share_5y: Optional[float] = None
    avg_by_category__amount__sum__cashflowcategory_name__platezhi_cherez_internet: Optional[float] = None
    hdb_bki_total_micro_max_overdue: Optional[float] = None
    bki_total_active_products: Optional[float] = None
    by_category__amount__sum__eoperation_type_name__perevod_s_karty_na_kartu: Optional[float] = None
    calledCtnOutGroup: Optional[float] = None
    vert_pil_loan_application_success_3m: Optional[float] = None
    vert_pil_fee_discount_change_3m: Optional[float] = None
    businessTelSubs: Optional[float] = None
    profit_income_out_rur_amt_l2m: Optional[float] = None
    avg_3m_healthcare_services: Optional[float] = None
    dp_ils_paymentssum_month_avg: Optional[float] = None
    ovrd_sum: Optional[float] = None
    hdb_bki_total_active_products: Optional[float] = None
    hdb_bki_total_micro_cnt: Optional[float] = None
    hdb_bki_active_pil_cnt: Optional[float] = None
    loan_cur_amt: Optional[float] = None
    mob_total_sessions: Optional[float] = None
    period_last_act_ad: Optional[float] = None
    dp_ils_days_multiple_job_share_2y: Optional[float] = None
    hdb_bki_total_cc_max_overdue: Optional[float] = None
    lifetimeComp: Optional[float] = None
    hdb_bki_total_pil_last_days: Optional[float] = None
    amount_by_category_90d__summarur_amt__sum__cashflowcategory_name__elektronnye_dengi: Optional[float] = None
    turn_save_cr_max_v2: Optional[float] = None
    hdb_bki_active_pil_max_limit: Optional[float] = None
    dp_ils_accpayment_avg_3m: Optional[float] = None
    avg_6m_restaurants: Optional[float] = None
    hdb_bki_total_pil_cnt: Optional[float] = None
    transaction_category_fastfood_percent_cnt_2m: Optional[float] = None
    hdb_bki_total_pil_max_del90: Optional[float] = None
    accountsalary_out_flag: Optional[float] = None
    cntBlockWavg6m: Optional[float] = None
    express_rur_amt_cm_avg: Optional[float] = None
    loanacc_rur_amt_cm_avg_inc_v2: Optional[float] = None
    hdb_bki_last_product_days: Optional[float] = None
    dp_ils_days_multiple_job_cnt_5y: Optional[float] = None
    dp_ils_accpayment_month_avg: Optional[float] = None
    cred_dda_rur_amt_3m_avg: Optional[float] = None
    avg_3m_all: Optional[float] = None
    hdb_other_active_max_psk: Optional[float] = None
    hdb_bki_other_active_ip_outstanding: Optional[float] = None
    total_sum: Optional[float] = None
    dp_ils_uniq_companies_1y: Optional[float] = None
    avg_6m_travel: Optional[float] = None
    avg_6m_government_services: Optional[float] = None
    hdb_bki_active_cc_max_overdue: Optional[float] = None
    total_rur_amt_cm_avg_period_days_ago_v2: Optional[float] = None
    label_Above_1M_share_r1: Optional[float] = None
    transaction_category_supermarket_sum_cnt_d15: Optional[float] = None
    max_balance_rur_amt_1m_af: Optional[float] = None
    w: Optional[float] = None
    first_salary_income: Optional[float] = None

def preprocess_single_client(client_data: Dict[str, Any], model_bundle) -> pd.DataFrame:
    scaler = model_bundle['scaler']
    features = model_bundle['features']
    medians = model_bundle['medians']
    upper_limits = model_bundle['upper_limits']
    pca = model_bundle['pca']
    maps = model_bundle['maps']

    df = pd.DataFrame([client_data])

    # Num fillna with train medians
    orig_num_cols = list(medians.keys())
    for col in orig_num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(medians[col])

    # Cat map with train maps
    cat_cols = list(maps.keys())
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("MISSING")
            df[col] = df[col].apply(lambda v: maps[col].get(v, maps[col].get("MISSING", -1)))

    # Feature eng
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

    # Clip with train upper
    for col in all_num_cols:
        if col in df.columns and col in upper_limits:
            df[col] = np.clip(df[col], None, upper_limits[col])

    # PCA transform if pca
    turnover_cols = [col for col in df.columns if 'turn_' in col]
    if pca and turnover_cols:
        pca_features = pca.transform(df[turnover_cols])
        pca_df = pd.DataFrame(pca_features, columns=[f'pca_turn_{i}' for i in range(5)])
        df = pd.concat([df, pca_df], axis=1)
        df.drop(columns=turnover_cols, inplace=True)

    # Update all_num_cols for pca
    pca_cols = [f'pca_turn_{i}' for i in range(5)] if turnover_cols else []
    all_num_cols = [col for col in all_num_cols if col not in turnover_cols] + pca_cols

    # Scale nums
    df[all_num_cols] = scaler.transform(df[all_num_cols])

    # Select features, fill missing with 0
    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]

@app.post("/predict_income")
async def predict_income(client: Dict[str, Any]):
    if model_bundle is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        X_client = preprocess_single_client(client, model_bundle)

        lgb_models = model_bundle['lgb_models']
        cat_models = model_bundle['cat_models']
        nn_models = model_bundle['nn_models']
        meta_model = model_bundle['meta_model']

        lgb_preds = np.mean([np.expm1(m.predict(X_client)) for m in lgb_models], axis=0)
        cat_preds = np.mean([np.expm1(m.predict(X_client)) for m in cat_models], axis=0)
        nn_preds = np.mean([np.expm1(m.predict(X_client.values, verbose=0).flatten()) for m in nn_models], axis=0)

        stack_preds = np.column_stack([lgb_preds, cat_preds, nn_preds])
        pred = np.expm1(meta_model.predict(stack_preds))[0]

        return {
            "predicted_income": float(pred),
            "model_version": "1.0.0"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/explain_income")
async def explain_income(client: ClientData):
    if model_bundle is None or explainer is None:
        raise HTTPException(status_code=500, detail="Model or explainer not loaded")

    try:
        client_dict = client.dict()
        X_client = preprocess_single_client(client_dict, model_bundle)

        lgb_models = model_bundle['lgb_models']
        cat_models = model_bundle['cat_models']
        nn_models = model_bundle['nn_models']
        meta_model = model_bundle['meta_model']

        lgb_preds = np.mean([np.expm1(m.predict(X_client)) for m in lgb_models], axis=0)
        cat_preds = np.mean([np.expm1(m.predict(X_client)) for m in cat_models], axis=0)
        nn_preds = np.mean([np.expm1(m.predict(X_client.values, verbose=0).flatten()) for m in nn_models], axis=0)

        stack_preds = np.column_stack([lgb_preds, cat_preds, nn_preds])
        pred = np.expm1(meta_model.predict(stack_preds))[0]

        # Compute SHAP values for the meta-model
        shap_values = explainer(stack_preds)

        # Extract SHAP explanation
        base_value = float(shap_values.base_values[0])
        shap_vals = shap_values.values[0].tolist()
        feature_names = ['LGB Prediction', 'CatBoost Prediction', 'NN Prediction']

        return {
            "predicted_income": float(pred),
            "shap_explanation": {
                "base_value": base_value,
                "shap_values": shap_vals,
                "feature_names": feature_names
            },
            "model_version": "1.0.0"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Explanation error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Income Prediction API", "docs": "/docs"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_bundle is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
