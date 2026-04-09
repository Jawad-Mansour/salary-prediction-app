# # scripts/quick_test.py
# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path.cwd()))

# print("=" * 60)
# print("COMPLETE VALIDATION TEST")
# print("=" * 60)

# # Test 1: Load data
# from salary_src.data_loader import load_salaries_dataset
# df = load_salaries_dataset()
# print(f"✅ 1. Data loaded: {df.shape}")

# # Test 2: Load encoding maps
# from salary_src.preprocess import load_encoding_maps, preprocess_single_row, get_full_feature_order
# encoding_maps = load_encoding_maps("models/encoding_maps.json")
# print(f"✅ 2. Encoding maps loaded")

# # Test 3: Get expected feature order
# expected_order = get_full_feature_order()
# print(f"✅ 3. Expected feature order: {len(expected_order)} features")

# # Test 4: Load model
# import joblib
# model = joblib.load("models/decision_tree.pkl")
# print(f"✅ 4. Model loaded: {type(model).__name__}")

# # Test 5: Check model features
# if hasattr(model, 'feature_names_in_'):
#     print(f"✅ 5. Model expects {len(model.feature_names_in_)} features")
#     if list(model.feature_names_in_) == expected_order:
#         print("   ✅ Feature order matches!")
#     else:
#         print("   ⚠️ Feature order mismatch - reordering will happen")
# else:
#     print("   ⚠️ Model has no feature_names_in_ attribute")

# # Test 6: Make prediction
# test_row = {
#     'experience_level': 'SE', 'employment_type': 'FT',
#     'job_title': 'Data Scientist', 'company_size': 'L',
#     'remote_ratio': 100, 'work_year': 2024,
#     'employee_residence': 'US', 'company_location': 'US'
# }

# X_input = preprocess_single_row(test_row, encoding_maps)
# print(f"✅ 6. Preprocessed input: {X_input.shape[1]} features")

# # Make prediction
# pred = model.predict(X_input)[0]
# print(f"✅ 7. Prediction: ${pred:,.2f}")

# # Test 7: Check metrics
# import json
# with open("models/metrics.json", 'r') as f:
#     metrics = json.load(f)
# print(f"✅ 8. Metrics: R²={metrics['r2']:.4f}, MAE=${metrics['mae']:,.2f}")

# print("\n" + "=" * 60)
# print("🎉 ALL TESTS PASSED! Ready for Phase 4.")
# print("=" * 60)