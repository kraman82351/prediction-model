import pandas as pds
from sklearn.model_selection import train_test_split as train_test_split
from sklearn.metrics import mean_squared_error as mean_squared_error
from sklearn.metrics import r2_score as r2_score
import xgboost as xgb
import joblib as joblib

def training_testing(df):

    # Create a new column with the count of ClaimIDs linked with each BeneID
    df['NumClaimsPerBene'] = df.groupby('BeneID')['ClaimID'].transform('count')

    # Convert 'ClaimStartDt' and 'ClaimEndDt' to datetime objects
    df['ClaimStartDt'] = pds.to_datetime(df['ClaimStartDt'])
    df['ClaimEndDt'] = pds.to_datetime(df['ClaimEndDt'])

    # Calculate the number of days to settle claims
    df['NumDaysToSettle'] = (df['ClaimEndDt'] - df['ClaimStartDt']).dt.days

    # Convert 'ClaimStartDt' and 'ClaimEndDt' to datetime objects
    df['AdmissionDt'] = pds.to_datetime(df['AdmissionDt'])
    df['DischargeDt'] = pds.to_datetime(df['DischargeDt'])

    # Calculate the number of days spent
    df['NumDaysSpent'] = (df['DischargeDt'] - df['AdmissionDt']).dt.days

    # Add a column for the count of non-null values in the claim diagnosis and claim Procedure code columns
    diagnosis_columns = [f'ClmDiagnosisCode_{i}' for i in range(1, 11)]
    procedure_columns = [f'ClmProcedureCode_{i}' for i in range(1, 7)]

    df['NumDiagnosisCodes'] = df[diagnosis_columns].notnull().sum(axis=1)
    df['NumProcedureCodes'] = df[procedure_columns].notnull().sum(axis=1)

    columns_to_exclude = ['ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
                        'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
                        'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10',
                        'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
                        'ClmProcedureCode_5', 'ClmProcedureCode_6',
                        'AdmissionDt', 'DischargeDt', 'ClaimStartDt', 'ClaimEndDt']

    # Create a new DataFrame excluding the specified columns
    new_df = df.drop(columns=columns_to_exclude)

    chronic_cond_columns = [
        'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
        'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
        'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
        'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke'
    ]

    # Convert values of 2 to 0(considering 1 is true and 2 is false)
    new_df[chronic_cond_columns] = new_df[chronic_cond_columns].replace(2, 0)

    # Calculate the sum of values across all ChronicCond columns for each BeneID
    new_df['TotalChronicCond'] = new_df[chronic_cond_columns].sum(axis=1)

    # Filter the BeneIDs where the sum is equal to the total number of ChronicCond columns
    num_chronic_diseases = new_df[new_df['TotalChronicCond'] == len(chronic_cond_columns)].groupby('BeneID').size()

    columns_to_exclude = ['DOD', 'DOB']

    # Create a new DataFrame excluding the specified columns
    new_df = new_df.drop(columns=columns_to_exclude)

    # Calculate the percentage of null values in each column
    null_percentage = (new_df.isnull().sum() / len(new_df)) * 100

    summary = pds.DataFrame({
        'Null Percentage': null_percentage,
        'Data Type': new_df.dtypes
    })

    # # Display the summary
    # print(summary)

    # Drop specified columns to create a new DataFrame
    new_df_filtered = new_df.drop(columns=['OperatingPhysician', 'OtherPhysician', 'ClmAdmitDiagnosisCode',
                                        'DiagnosisGroupCode', 'NumDaysSpent'])


    # Calculate the percentage of null values in each column
    null_percentage = (new_df_filtered.isnull().sum() / len(new_df_filtered)) * 100

    summary = pds.DataFrame({
        'Null Percentage': null_percentage,
        'Data Type': new_df_filtered.dtypes
    })

    new_df_filtered = new_df_filtered.dropna(subset=['DeductibleAmtPaid'])
    new_df_filtered = new_df_filtered.dropna(subset=['AttendingPhysician'])

    from sklearn.model_selection import train_test_split as train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib

    # Function to load previous model from the model.pkl file
    def load_previous_model(file_path):
        try:
            prev_model = joblib.load(file_path)
            print("Previous model is loaded")
            return prev_model
        except FileNotFoundError:
            return None

    def load_previous_results(file_path):
        try:
            prev_results = pds.read_csv(file_path)
            return prev_results
        except FileNotFoundError:
            return None

    # Define independent and dependent variables
    X = new_df_filtered.drop(columns=['BeneID', 'ClaimID', 'InscClaimAmtReimbursed', 'PotentialFraud', 'RenalDiseaseIndicator',
                                    'AttendingPhysician', 'Provider', 'NoOfMonths_PartBCov', 'NoOfMonths_PartACov',
                                    'ChronicCond_IschemicHeart', 'ChronicCond_Diabetes', 'ChronicCond_Heartfailure',
                                    'ChronicCond_Cancer', 'ChronicCond_KidneyDisease'])

    y = new_df_filtered['InscClaimAmtReimbursed']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Load previous model from model.pkl
    prev_model = load_previous_model('model.pkl')

    if prev_model is not None:
        # Load previous results from results.csv
        print("Testing using the previous model")
        prev_results = load_previous_results('results.csv')

        # Make predictions using previous model
        y_pred_train = prev_model.predict(X_train)
        y_pred_test = prev_model.predict(X_test)

        # Calculate R-squared for training and testing sets
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        prev_results['Training R-squared'] = pds.to_numeric(prev_results['Training R-squared'], errors='coerce')
        prev_results['Testing R-squared'] = pds.to_numeric(prev_results['Testing R-squared'], errors='coerce')

        if prev_results is None or (r2_train - prev_results['Training R-squared'].iloc[-1]) >= 0.05 \
                or (r2_test - prev_results['Testing R-squared'].iloc[-1]) >= 0.05:
            # Save the current results to a CSV file
            current_results = pds.DataFrame({
                'Training RMSE': [prev_results['Training RMSE'].iloc[-1]],
                'Testing RMSE': [prev_results['Testing RMSE'].iloc[-1]],
                'Training R-squared': [r2_train],
                'Testing R-squared': [r2_test]
            })

            print("Result Saved successfully")
            # Save the current results and training data to a CSV file
            with open('results.csv', 'a') as f:
                current_results.to_csv(f, mode='a', index=False, header=f.tell() == 0)
            if prev_results is None:
                X_train.to_csv('improved_results_X_train.csv', index=False)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Initialize and train the xgbsoost Regression model
            xgbs_model = xgb.xgbsRegressor(objective ='reg:squarederror', n_estimators=100, random_state=42)
            xgbs_model.fit(X_train, y_train)

            # Save the new model
            joblib.dump(xgbs_model, 'model.pkl')

            # Make predictions
            y_pred_train = xgbs_model.predict(X_train)
            y_pred_test = xgbs_model.predict(X_test)

            # Calculate training and testing RMSE
            rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
            rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)

            # Calculate R-squared for training and testing sets
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)

            # Save the current results to a CSV file
            current_results = pds.DataFrame({
                'Training RMSE': [rmse_train],
                'Testing RMSE': [rmse_test],
                'Training R-squared': [r2_train],
                'Testing R-squared': [r2_test]
            })
            print("Result Saved successfully")
            current_results.to_csv('results.csv', mode='a', index=True, header=True)

    else:
        # Split data into training and testing sets
        print("Previous Model accuracy is less so new model is trained and tested")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize and train the xgbsoost Regression model
        xgbs_model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=100, random_state=42)
        xgbs_model.fit(X_train, y_train)

        # Save the new model
        joblib.dump(xgbs_model, 'model.pkl')

        # Make predictions
        y_pred_train = xgbs_model.predict(X_train)
        y_pred_test = xgbs_model.predict(X_test)

        # Calculate training and testing RMSE
        rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
        rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)

        # Calculate R-squared for training and testing sets
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        # Save the current results to a CSV file
        current_results = pds.DataFrame({
            'Training RMSE': [rmse_train],
            'Testing RMSE': [rmse_test],
            'Training R-squared': [r2_train],
            'Testing R-squared': [r2_test]
        })
        print("Result Saved successfully")
        current_results.to_csv('results.csv', mode='a', index=True, header=True)

    # # Print the current results
    # print("Training RMSE:", rmse_train)
    # print("Testing RMSE:", rmse_test)
    # print("Training R-squared:", r2_train)
    # print("Testing R-squared:", r2_test)
