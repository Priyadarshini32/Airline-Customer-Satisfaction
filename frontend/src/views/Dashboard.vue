<template>
  <div class="dashboard-container">
    <div :class="['dashboard-form', step === 1 ? 'step-1' : 'step-2']">
      <div v-if="authStore.isLoggedIn" class="satisfaction-form pa-5">
        <div v-if="step === 1" class="form-step">
          <div class="pb-3">
            <h3 class="step-title">Step 1: Provide your details</h3>
          </div>
          <div class="form-group">
            <label for="Gender">Gender</label>
            <select id="Gender" v-model="formData.Gender" required>
              <option value="">Select</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </div>
          <div class="form-group">
            <label for="Customer Type">Customer Type</label>
            <select
              id="Customer Type"
              v-model="formData['Customer Type']"
              required
            >
              <option value="">Select</option>
              <option value="Frequent Traveler">Frequent Traveler</option>
              <option value="Occasional Traveler">Occasional Traveler</option>
              <option value="Rare Traveler">Rare Traveler</option>
            </select>
          </div>
          <div class="form-group">
            <label for="Age">Age</label>
            <input
              type="number"
              id="Age"
              v-model="formData.Age"
              required
              min="0"
              max="120"
            />
          </div>
          <div class="form-group">
            <label for="Type of Travel">Type of Travel</label>
            <select
              id="Type of Travel"
              v-model="formData['Type of Travel']"
              required
            >
              <option value="">Select</option>
              <option value="Business travel">Business Travel</option>
              <option value="Personal Travel">Personal Travel</option>
            </select>
          </div>
          <div class="form-group">
            <label for="Class">Class</label>
            <select id="Class" v-model="formData.Class" required>
              <option value="">Select</option>
              <option value="Eco">Economy</option>
              <option value="Eco Plus">Economy Plus</option>
              <option value="Business">Business</option>
            </select>
          </div>
          <div class="form-group">
            <label for="Flight Distance">Flight Distance</label>
            <input
              type="number"
              id="Flight Distance"
              v-model="formData['Flight Distance']"
              required
              min="0"
            />
          </div>
          <button @click="nextStep" class="next-btn">Next</button>
        </div>

        <div v-if="step === 2" class="form-step">
          <h3 class="step-title">Step 2: Service Ratings</h3>
          <div class="rating-grid">
            <div class="rating-column">
              <div
                class="form-group"
                v-for="feature in firstColumnFeatures"
                :key="feature"
              >
                <label :for="feature">{{ feature }}</label>
                <RatingComponent v-model="formData[feature]" />
              </div>
            </div>
            <div class="rating-column">
              <div
                class="form-group"
                v-for="feature in secondColumnFeatures"
                :key="feature"
              >
                <label :for="feature">{{ feature }}</label>
                <RatingComponent v-model="formData[feature]" />
              </div>
            </div>
          </div>
          <div class="d-flex justify-space-between">
            <button @click="prevStep" class="next-btn">Back</button>
            <button
              @click="predictSatisfaction"
              class="submit-btn"
              :disabled="isPredictionMade"
            >
              Predict Satisfaction
            </button>
          </div>
        </div>
        <div v-if="showModal" class="modal-overlay">
          <div class="modal-box">
            <h3>Prediction Result</h3>
            <!-- Show Prediction Result -->
            <div v-if="isPredictionMade" class="prediction-result pa-3">
              <p>Predicted Satisfaction: {{ authStore.prediction }}</p>
              <button @click="resetForm" class="mt-8 submit-btn">
                New Prediction
              </button>
            </div>
          </div>
        </div>
      </div>
      <div v-else class="auth-warning">
        <h4>Please log in to access the prediction.</h4>
      </div>

      <div v-if="authStore.errorMessage" class="error-message">
        {{ authStore.errorMessage }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from "vue";
import { useAuthStore } from "@/stores/store";
import RatingComponent from "@/components/RatingComponent.vue";

const authStore = useAuthStore();
const step = ref(1);
const isPredictionMade = ref(false);
const showModal = ref(false);
const formData = ref({
  Gender: "",
  "Customer Type": "",
  Age: "",
  "Type of Travel": "",
  Class: "",
  "Flight Distance": "",
});

onMounted(async () => {
  await authStore.fetchSelectedFeatures();
  authStore.ratingFeatures.forEach((feature) => {
    formData.value[feature] = "";
  });
});

const firstColumnFeatures = computed(() =>
  authStore.ratingFeatures.slice(0, 5)
);
const secondColumnFeatures = computed(() => authStore.ratingFeatures.slice(5));

const nextStep = () => {
  step.value = 2;
};

const prevStep = () => {
  step.value = 1;
};

const predictSatisfaction = async () => {
  if (!authStore.isLoggedIn) {
    authStore.errorMessage = "You have been logged out.";
    return;
  }

  if (isPredictionMade.value) {
    return;
  }

  const hasEmptyFields = Object.values(formData.value).some(
    (value) => value === ""
  );
  if (hasEmptyFields) {
    authStore.errorMessage = "Please fill out all required fields.";
    return;
  }

  authStore.errorMessage = null;
  formData.value["Customer Type"] = mapCustomerType(
    formData.value["Customer Type"]
  );

  await authStore.fetchPrediction(formData.value);
  isPredictionMade.value = true;
  showModal.value = true;
};

const resetForm = () => {
  showModal.value = false;
  step.value = 1;
  isPredictionMade.value = false;
  formData.value = {
    Gender: "",
    "Customer Type": "",
    Age: "",
    "Type of Travel": "",
    Class: "",
    "Flight Distance": "",
  };
  authStore.ratingFeatures.forEach((feature) => {
    formData.value[feature] = "";
  });
};

const mapCustomerType = (customerType) => {
  switch (customerType) {
    case "Frequent Traveler":
      return "Loyal Customer";
    case "Occasional Traveler":
      return "Loyal Customer";
    case "Rare Traveler":
      return "disloyal Customer";
    default:
      return "";
  }
};
</script>

<style scoped>
.dashboard-container {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 87%;
  font-family: "Trebuchet MS", "Lucida Sans Unicode", "Lucida Grande",
    "Lucida Sans", Arial, sans-serif;
  font-weight: bold;
  margin-left: 7%;
  margin-top: 0.9%;
  color: rgb(48, 47, 47);
}

.dashboard-form {
  padding: 10px;
  background-color: #fff;
  border-radius: 10px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  text-align: center;
  box-sizing: border-box;
}

/* Step-specific width adjustments */
.dashboard-form.step-1 {
  width: 700px; /* Step 1 container width */
}

.dashboard-form.step-2 {
  width: 900px; /* Step 2 container width */
}

.form-group {
  display: flex;
  align-items: center;
  margin-bottom: 20px;
  position: relative;
  padding-bottom: 10px;
}

.form-group label {
  flex-basis: 30%;
  text-align: center;
  margin-right: 10px;
  font-weight: bold;
  color: #2b929c;
  font-size: large;
}

.form-group select,
.form-group input {
  flex: 1;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 5px;
  font-size: 16px;
  background-color: #f9f9f9;
  transition: border-color 0.3s ease;
}

.form-group select:focus,
.form-group input:focus {
  outline: none;
  border-color: #42b983;
  background-color: #fff;
}

.submit-btn {
  background-color: #42b983;
  color: #fff;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 17px;
  transition: background-color 0.3s ease;
  padding: 10px;
  width: 30%;
}

.next-btn {
  background-color: #828c87;
  color: #fff;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 17px;
  transition: background-color 0.3s ease;
  padding: 10px;
  width: 20%;
}

.submit-btn:hover {
  background-color: #d72863;
}
.next-btn:hover {
  background-color: #e5b72e;
}

.rating-grid {
  display: flex;
  justify-content: space-between;
  gap: 10px;
}

.rating-column {
  flex: 1;
}

.rating-star {
  display: inline-block;
  font-size: 24px;
  color: #ddd;
  cursor: pointer;
}

.rating-star.filled {
  color: #f39c12;
}

.error-message {
  color: red;
  font-size: 14px;
}

/* Modal styles */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal-box {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  width: 400px;
  text-align: center;
}

.auth-warning {
  font-size: 19px;
  color: #e58961;
}

.step-title {
  color: #287b51; /* Grey color */
  font-size: 1.2rem;
  font-weight: bold;
}
</style>
