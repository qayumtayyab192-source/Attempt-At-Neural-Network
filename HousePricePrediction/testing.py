from pathlib import Path
from sklearn.datasets import fetch_california_housing

path = Path(__file__).parent

OverwritingFile = path / "HousePricePredictor" / "HouseInformation.txt"
ActualPrices = path / "HousePricePredictor" / "ActualPrices.txt"

housing = fetch_california_housing(as_frame=True)
X = housing.data.values.tolist()
Y = housing.target.values.tolist()

with open(OverwritingFile, "w") as f:
    for i in range(20639):
        for j in range(8):
            f.write(str(X[i][j]))
        f.write("\n")

with open(ActualPrices, "w") as f:
    for i in range(20639):
        f.write(str(Y[i]))
        f.write("\n")

