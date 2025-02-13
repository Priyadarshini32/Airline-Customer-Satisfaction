<template>
  <div class="login-container">
    <div class="login-form elevation-4">
      <h2>Login</h2>
      <p>Please enter your credentials to log in.</p>
      <form @submit.prevent="handleLogin">
        <div class="form-group mt-5">
          <label for="username">Username</label>
          <div class="input-wrapper">
            <input type="text" id="username" v-model="username" required />
          </div>
        </div>

        <div class="form-group">
          <label for="password">Password</label>
          <div class="input-wrapper">
            <input type="password" id="password" v-model="password" required />
          </div>
        </div>

        <p v-if="errorMessage" class="error">{{ errorMessage }}</p>

        <button type="submit">Login</button>
      </form>
      <p class="register-link">
        Don't have an account?
        <router-link to="/register">Register here</router-link>
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { useAuthStore } from "@/stores/store";
import { useRouter } from "vue-router";

const authStore = useAuthStore();
const router = useRouter();
const username = ref("");
const password = ref("");
const errorMessage = ref("");

const handleLogin = async () => {
  const success = await authStore.login(username.value, password.value);
  console.log("helloo", success);
  if (success) {
    router.push("/dashboard"); // Redirect after login
  } else {
    errorMessage.value = authStore.errorMessage;
  }
};
</script>

<style scoped>
.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 87%;
  font-family: "Trebuchet MS", "Lucida Sans Unicode", "Lucida Grande",
    "Lucida Sans", Arial, sans-serif;
  font-weight: bold;
  margin-left: 7%;
  margin-top: 6%;
  color: rgb(48, 47, 47);
}

.login-form {
  max-width: 500px;
  padding: 40px;
  background-color: #fff;
  border-radius: 10px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  text-align: center;
  box-sizing: border-box;
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
  text-align: right;
  margin-right: 10px;
  font-weight: 550;
  color: #2b929c;
  font-size: large;
}

.input-wrapper {
  flex: 1;
  display: flex;
  align-items: center;
  position: relative;
}

.input-wrapper input {
  width: 100%;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 5px;
  font-size: 15px;
  background-color: #f9f9f9;
  transition: border-color 0.3s ease;
}

.input-wrapper input:focus {
  outline: none;
  border-color: #42b983;
  background-color: #fff;
}

.error {
  color: red;
  font-size: 14px;
  margin-bottom: 10px;
}

button {
  background-color: #42b983;
  color: #fff;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 17px;
  transition: background-color 0.3s ease;
  padding: 10px;
  width: 40%;
}

button:hover {
  background-color: #35495e;
}

.register-link {
  margin-top: 20px;
  font-size: 16px;
  color: #787777;
}

.register-link a {
  color: #42b983;
  text-decoration: none;
}

.register-link a:hover {
  text-decoration: underline;
}
</style>
