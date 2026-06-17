# Shipped model artifacts

This directory holds the committed ONNX model artifact served at inference time
(`lstm_forecast.onnx`, exported by `lstmforecast.models.lstm.export_onnx` and
loaded by `lstmforecast.models.onnx_runtime.OnnxForecaster`).

The shipped model is trained on **synthetic random-walk data** (see
`lstmforecast.data.random_walk_prices`) — there is no real market data or API key
in this repo. On a random walk the next-day return is unpredictable, so the
honest NULL (the LSTM does **not** beat persistence) holds by construction and the
artifact is fully reproducible. Retrain on real data via `lstm-forecast train
--data <csv>`.

`*.onnx` files are **committed** (they ship inside the wheel); `*.h5`/`*.pkl`
intermediates are git-ignored.
