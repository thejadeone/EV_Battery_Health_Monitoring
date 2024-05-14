import streamlit as st
# import utils.preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib


def data_encoder(batteryData):
    # from sklearn.preprocessing import StandardScaler
    batterydf = pd.DataFrame.from_dict(batteryData)
    predictorScaler = StandardScaler()

    predictorScalerFit = predictorScaler.fit_transform(batterydf)
    randomforest = joblib.load(open('./notebook/BatteryHealthPredictorModel.pkl', 'rb'))
    prediction = randomforest.predict(predictorScalerFit)
    return prediction


def main():
    with st.container(border=True):
        st.title("Battery Health Monitor using ML")
        with st.container(border=True):
            # Columns
            # Voltage
            # current
            # temperature
            # Capacity
            batteryData = {}

            voltage = st.number_input("Insert the voltage (Volts)",format="%.4f", value=None, placeholder="Type a number...")
            st.write('The current number is ', voltage)
            batteryData['voltage'] = [voltage]

            current = st.number_input("Insert current (Amp)",format="%.4f", value=None, placeholder="Type a number...")
            st.write('The current number is ', current)
            batteryData['current'] = [current]

            temperature = st.number_input("Insert temperature (C)",format="%.4f", value=None, placeholder="Type a number...")
            st.write('The current number is ', temperature)
            batteryData['temperature'] = [temperature]

            capacity = st.number_input("Insert the capacity",format="%.4f", value=None, placeholder="Type a number...")
            st.write('The current number is ', capacity)
            batteryData['capacity'] = [capacity]
            batteryData = batteryData.round(
                {'voltage': 4, 'current': 4, 'temperature': 4, 'capacity': 4 })

            # userData = {}

            with st.container(border=True):
                if st.button("Predict"):
                    print('Battery Data -', batteryData)
                    with st.container(border=True):
                        prediction = data_encoder(batteryData)
                        print('prediction - ', prediction)
                        if prediction[0]:
                            st.subheader("SOC Capacity : ")
                            st.markdown(prediction[0])


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
