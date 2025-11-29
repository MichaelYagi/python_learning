# Simple Linear Regression from scratch
# We compute the slope and intercept using the least-squares formula.
#
# The model is
# 𝑦 = 𝑚 ⋅ 𝑥 + 𝑏
# We then predict a new value (exam score for 6 hours of study).

# Training data: x = study hours, y = exam score
X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]
print("X: ", X)
print("Y: ", Y)

# Step 1: Calculate averages
x_mean = sum(X) / len(X)
y_mean = sum(Y) / len(Y)
print("x_mean: ", x_mean)
print("y_mean: ", y_mean)

# Step 2: Calculate slope (m)
# We want to fit a straight line: 𝑦 = 𝑚 ⋅ 𝑥 + 𝑏
# that best matches the data points (𝑋, 𝑌)
# Each term (𝑋[𝑖] − 𝑥 mean) is how far the 𝑖-th 𝑥 value is from the average.
# Each term (𝑌[𝑖] − 𝑦 mean) is how far the 𝑖-th 𝑦 value is from the average.
# Multiplying them together measures how much 𝑥 and 𝑦 vary together (covariance).
# Summing over all points gives the total covariance between 𝑋 and 𝑌.
# num is essentially the strong covariance between 𝑋 and 𝑌 (data points closest to the line).
num = sum((X[i] - x_mean) * (Y[i] - y_mean) for i in range(len(X)))
den = sum((X[i] - x_mean) ** 2 for i in range(len(X)))
m = num / den
print("slope m: ", m)

# Step 3: Calculate intercept (b)
b = y_mean - m * x_mean
print("intercept b: ", b)

print("Model: y =", round(m, 2), "* x +", round(b, 2))

# Step 4: Make predictions
def predict(x):
    return m * x + b

print("Prediction for 6 study hours:", predict(6))
