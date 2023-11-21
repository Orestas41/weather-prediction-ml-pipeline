# Weather Predictor

## Model Details

- Developer: Orestas Dulinskas
- Model date: 27/11/2023
- Model Version: 1.0.0
- Model type: XGBRegressor and Random Forest Classifier from scikit-learn

## Model Description

Weather predictor predicts weather features such as maximum and minimum temperature, precipitation sum and weather code (WMO code 4667) that describes expected weather of the day. It uses a combination of models to output all predictions:

### XGBRegressor 

This model is used to predict temperature and precipitation as float values. It takes date and location values as input and outputs maximum and minimum temperatures as degrees Celsius, and precipitation sum as millimeters.

### Random Forest Classifier 

This model predicts the weather code as an integer. As an input, it takes date and location, as well as, the outputs of temperature and precipitation prediction. Then it outputs an integer between 0 and 99 that represents a weather description from WMO code 4677.

## Model Parameters

### XGBRegressor

- learning_rate: 0.095
- max_depth: 9
- subsample: 0.95
- n_estimators: 150
- gamma: 0.3
- reg_alpha: 0.43

### Random Forest Classifier

- min_samples_split: 2
- min_samples_leaf: 1
- n_estimators: 325
- criterion: 'gini'
- max_features: 'sqrt'
- bootstrap: True

## Model Training Data

The model has been trained on 15 years (2008-2023) worth of daily weather data of 10 major cities in UK. It was pulled from Historical Weather API on Open-Meteo (https://open-meteo.com/). Data consisted of the date, weather code, max and min temperatures and precipitation sum. Training data is split into 49% training data, 21% validation data and 30% testing data.

### Pre-processing:

- Creating month-day column. This is a float value with whole digits corresponding to the month and decimals corresponding to the day. This have been feature engineered using the date values, and have been proven to improve model performance.
- Set date as index
- Drop Na and duplicates
- Sort data by date

## Model Evaluation Metrics

The model's performance is assessed using Mean Absolute Error (MAE) and R-squared (R2) scores. MAE measures the average absolute difference between the predicted and actual outcomes, while R2 indicates the proportion of the variance in the target variable that is predictable. A lower MAE value and a higher R2 value signify better model performance.

### Model performance measures:

#### XGBRegressor
- MAE = 2.54
- R2 = 0.44

#### Random Forest Classifier
- MAE = 1.37
- R2 = 0.69

## Model Limitations

### Input Variability
The model heavily relies on input features such as date, location, and historical weather attributes. Variations in input data quality or missing relevant features may impact its accuracy. Factors not accounted for in the training data, such as local geographical features or extreme weather events, could affect its predictions.

### Dynamic Nature of Weather
Weather prediction involves complex, dynamic systems influenced by numerous variables. The model's static nature might struggle to adapt to abrupt changes or rare weather phenomena not adequately represented in the training data, potentially leading to inaccuracies.

### Assumed Dependencies
The model assumes certain dependencies between weather features, which might not always hold true in real-world scenarios. For instance, it might overlook intricate relationships between various atmospheric conditions, impacting the accuracy of its predictions, such as 

## Model Retraining

To enhance and maintain performance, the model will undergo weekly retraining. This process involves pulling the latest data from the API, concatenating it with the existing data, and retraining the model. If the model demonstrates improved performance, it will be automatically promoted for production use.

## Ethical Considerations

Weather predictions play a crucial role in various sectors, including agriculture, transportation, and emergency services. If used in critical decision-making processes, it's important to acknowledge the limitations and uncertainties of the model to prevent over-reliance and potential misinterpretation of its outputs.
It is not recommended to use this model for any serious decision-making processes.

## Model Contact Information

For any inquiries or feedback related to the Weather Predictor model, please contact: orestasdulinskas@gmail.com.