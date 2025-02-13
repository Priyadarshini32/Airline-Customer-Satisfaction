import axios from "axios";
import { defineStore } from "pinia";
import { ref, computed } from "vue";

const baseUrl = "http://127.0.0.1:5000";

export const useAuthStore = defineStore("auth", () => {
  const user = ref(null);
  const errorMessage = ref(null);
  const prediction = ref(null);
  const staticFeatures = ref([]);
  const ratingFeatures = ref([]);
  const debugInfo = ref(null);
  const predictions = ref(null);
  // Register action
  const register = async (username, email, password) => {
    try {
      const response = await axios.post(`${baseUrl}/register`, {
        username,
        email,
        password,
      });

      if (response.status === 201) {
        alert("Registration Successful! Please log in.");
        return true;
      }
    } catch (error) {
      errorMessage.value =
        error.response?.data?.error || "Registration failed!";
      return false;
    }
  };

  // Login action
  const login = async (username, password) => {
    try {
      const response = await axios.post(`${baseUrl}/login`, {
        username,
        password,
      });
      if (response.status === 200) {
        user.value = response.data.username;
        localStorage.setItem("user", JSON.stringify(user.value));
        return true;
      }
    } catch (error) {
      errorMessage.value = error.response?.data?.error || "Login failed!";
      return false;
    }
  };

  // Fetch selected features
  const fetchSelectedFeatures = async () => {
    try {
      const response = await axios.get(`${baseUrl}/selected_features`);

      if (response.status === 200) {
        staticFeatures.value = response.data.static_features || [];
        ratingFeatures.value = response.data.selected_rating_features || [];
      }
    } catch (error) {
      errorMessage.value =
        "Error fetching features: " +
        (error.response?.data?.error || error.message);
      console.error("Feature fetch error:", error);
    }
  };

  // Fetch prediction result and store it
  const fetchPrediction = async (formData) => {
    try {
      prediction.value = null;
      errorMessage.value = null;
      debugInfo.value = null;

      // Process form data
      const processedData = {
        ...formData,
        Age: Number(formData.Age),
        "Flight Distance": Number(formData["Flight Distance"]),
      };

      // Convert rating features to numbers
      ratingFeatures.value.forEach((feature) => {
        if (formData[feature]) {
          processedData[feature] = Number(formData[feature]);
        }
      });

      // Get prediction
      const response = await axios.post(`${baseUrl}/predict`, processedData);

      if (response.status === 200) {
        prediction.value = response.data.prediction;
        debugInfo.value = response.data.debug_info;

        // Store the prediction
        await storePrediction(processedData, prediction.value);
      }
    } catch (error) {
      console.error("Prediction error:", error);
      errorMessage.value =
        error.response?.data?.error || "Error making prediction";
    }
  };

  // Store prediction in database
  const storePrediction = async (formData, predictionResult) => {
    const response = await axios.post(`${baseUrl}/store_prediction`, {
      username: user.value,
      form_data: formData,
      prediction: predictionResult,
    });

    if (response.status !== 201) {
      console.error("Failed to store prediction");
      errorMessage.value = "Failed to store prediction";
    }
  };
  // // Load last prediction from localStorage
  // const loadLastPrediction = () => {
  //   const lastPrediction = localStorage.getItem("last_prediction");
  //   if (lastPrediction) {
  //     const predictionData = JSON.parse(lastPrediction);
  //     prediction.value = predictionData.prediction;
  //     accuracy.value = predictionData.accuracy;
  //     debugInfo.value = predictionData.debug_info;
  //     console.log("Loaded last prediction:", predictionData);
  //   }
  // };

  // const clearPrediction = () => {
  //   prediction.value = null;
  //   accuracy.value = null;
  //   debugInfo.value = null;
  //   errorMessage.value = null;
  // };
  const fetchAllPredictions = async () => {
    try {
      if (!user.value) {
        errorMessage.value = "User is not logged in.";
        return;
      }

      const response = await axios.get(`${baseUrl}/get_predictions`, {
        params: { username: user.value },
      });
      console.log("info", response);

      if (response.status === 200) {
        predictions.value = response.data; // Store all predictions
      } else {
        errorMessage.value = "No predictions available";
      }
    } catch (error) {
      errorMessage.value =
        error.response?.data?.error || "Failed to fetch predictions";
    }
  };
  // Load user from localStorage (on page refresh)
  const loadUserFromStorage = () => {
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      user.value = JSON.parse(storedUser);
    }
  };

  const logout = () => {
    user.value = null;
    localStorage.removeItem("user");
  };
  // Load user on startup
  loadUserFromStorage();

  const isLoggedIn = computed(() => !!user.value);

  return {
    user,
    errorMessage,
    prediction,
    predictions,
    staticFeatures,
    ratingFeatures,
    debugInfo,
    register,
    login,
    logout,
    fetchAllPredictions,
    loadUserFromStorage,
    fetchSelectedFeatures,
    fetchPrediction,
    storePrediction,
    isLoggedIn,
  };
});
