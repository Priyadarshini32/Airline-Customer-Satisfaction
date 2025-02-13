<template>
  <div class="prediction-page">
    <h1>Prediction Results</h1>

    <div v-if="predictions.length > 0">
      <h2>All Predictions</h2>
      <table class="prediction-table">
        <thead>
          <tr>
            <th>S.No</th>
            <th>Prediction</th>
            <th>Form Data</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(prediction, index) in predictions" :key="index">
            <td>{{ index + 1 }}</td>
            <td>{{ prediction.prediction }}</td>
            <td>
              <ul>
                <li v-for="(value, key) in prediction.form_data" :key="key">
                  <strong>{{ key }}:</strong> {{ value }}
                </li>
              </ul>
            </td>
            <td>{{ formatTimestamp(prediction.timestamp) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <p v-else>No predictions found.</p>
  </div>
</template>

<script>
import { onMounted, computed } from "vue";
import { useAuthStore } from "@/stores/store";

export default {
  name: "PredictionPage",
  setup() {
    const store = useAuthStore();

    // Fetch predictions when the component is mounted
    onMounted(() => {
      store.fetchAllPredictions();
    });

    return {
      // Default predictions as empty array
      predictions: computed(() => store.predictions || []), // Set default as empty array
      formatTimestamp: (timestamp) => {
        return new Date(timestamp).toLocaleString(); // Format timestamp
      },
    };
  },
};
</script>

<style scoped>
.prediction-page {
  max-width: 900px;
  margin: auto;
  padding: 20px;
}

h1,
h2 {
  text-align: center;
}

.prediction-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 20px;
}

.prediction-table th,
.prediction-table td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: left;
  vertical-align: top;
}

.prediction-table th {
  background-color: #f4f4f4;
}

ul {
  margin: 0;
  padding-left: 20px;
}
</style>

<style scoped>
.prediction-page {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
  font-family: "Trebuchet MS", "Lucida Sans Unicode", "Lucida Grande",
    "Lucida Sans", Arial, sans-serif;
}

h1,
h2,
h3 {
  font-size: 1.5rem;
  color: #333;
}

.prediction-info,
.prediction-form,
.prediction-result {
  margin-top: 1.5rem;
}

.prediction-info p,
.prediction-result p {
  font-size: 1rem;
  color: #444;
}

.prediction-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 1rem;
  text-align: left;
}

.prediction-table th,
.prediction-table td {
  padding: 10px;
  border: 1px solid #ddd;
}

.prediction-table th {
  background-color: #f5f5f5;
  font-weight: bold;
}

.prediction-form label {
  display: block;
  margin: 0.5rem 0;
}

.prediction-form input {
  padding: 0.5rem;
  margin: 0.5rem 0;
  width: 100%;
  border: 1px solid #ddd;
}

button {
  padding: 0.5rem 1rem;
  background-color: #42b983;
  color: white;
  border: none;
  cursor: pointer;
  font-size: 18px;
  transition: background-color 0.3s ease;
}

button:hover {
  background-color: #35495e;
}

.error {
  color: red;
}

.prediction-result {
  margin-top: 2rem;
  padding: 1rem;
  background-color: #f9f9f9;
  border: 1px solid #ddd;
}
</style>
