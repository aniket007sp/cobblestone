import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Objective 1: Algorithm Selection
def select_algorithm():
    return IsolationForest(contamination=0.02)  # Adjust parameters based on requirements

# Objective 2: Data Stream Simulation
def simulate_data_stream(num_points=1000, noise_level=0.1):
    normal_pattern = np.sin(np.arange(num_points) * 0.02)  # Example regular pattern
    random_noise = noise_level * np.random.randn(num_points)
    data_stream = normal_pattern + random_noise
    return data_stream.reshape(-1, 1)  # Reshape the data stream to a 2D array

# Objective 3: Anomaly Detection
def detect_anomalies(data_point, model):
    anomaly_score = model.predict(data_point)
    return anomaly_score

# Objective 5: Visualization
def visualize_stream_and_anomalies(data_stream, anomaly_indices):
    fig, ax = plt.subplots()

    def update(frame):
        ax.cla()
        ax.plot(data_stream[:frame + 1], label="Data Stream")

        if frame in anomaly_indices:
            ax.scatter(frame, data_stream[frame], color='red', label='Anomaly')

        ax.legend()
        ax.set_title(f"Data Stream and Anomalies (Frame {frame})")

    ani = FuncAnimation(fig, update, frames=len(data_stream), repeat=False)
    plt.show()


# Objective 1: Algorithm Selection
algorithm = select_algorithm()

# Objective 2: Data Stream Simulation
data_stream = simulate_data_stream()

# Objective 3: Anomaly Detection
algorithm.fit(data_stream)  # Fit the Isolation Forest model with the data

anomalies = []  # List to store anomaly indices
for frame, data_point in enumerate(data_stream):
    anomaly_score = detect_anomalies(data_point.reshape(1, -1), algorithm)
    if anomaly_score < 0:
        anomalies.append(frame)  # Store indices of anomalies

print(anomalies)
# Objective 5: Visualization
visualize_stream_and_anomalies(data_stream, anomalies)