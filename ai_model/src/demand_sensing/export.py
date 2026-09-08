import joblib
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType, FloatTensorType
from skl2onnx import update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from .features import CATEGORICAL_COLS

def export_onnx_model(model_pkl_path="output/demand_model_multi.pkl", encoder_pkl_path="output/categorical_encoder.pkl", output_onnx_path="output/agent_demand_multi.onnx"):
    model = joblib.load(model_pkl_path)
    encoder = joblib.load(encoder_pkl_path)

    # MultiOutputRegressor contains the underlying LightGBM models in `estimators_`
    base_model = model.estimators_[0]
    all_features = base_model.feature_name_
    numerical_cols = [col for col in all_features if col not in CATEGORICAL_COLS]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', encoder, CATEGORICAL_COLS),
            ('num', 'passthrough', numerical_cols)
        ]
    )
    preprocessor.transformers_ = preprocessor.transformers
    preprocessor.sparse_output_ = False

    local_agent_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    input_signature = []
    for col in all_features:
        if col in CATEGORICAL_COLS:
            input_signature.append((col, StringTensorType([None, 1])))
        else:
            input_signature.append((col, FloatTensorType([None, 1])))

    print("Registering custom LightGBM converter nodes to compiler backend...")
    try:
        update_registered_converter(
            lgb.LGBMRegressor, 
            'LightGbmLGBMRegressor',
            calculate_linear_regressor_output_shapes, 
            convert_lightgbm
        )
    except ValueError:
        pass

    print("Compiling unified structural ONNX graph with native column mapping opset...")
    onnx_graph = convert_sklearn(
        local_agent_pipeline, 
        name="agent_demand_multi", 
        initial_types=input_signature,
        target_opset={"": 15, "ai.onnx.ml": 2}
    )

    with open(output_onnx_path, "wb") as f:
        f.write(onnx_graph.SerializeToString())

    print(f"Success! Unified ONNX binary compiled successfully: {output_onnx_path}")

if __name__ == "__main__":
    export_onnx_model()
