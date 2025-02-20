<template>
  <div>
    <nav class="navbar">
      <div class="navbar-container">
        <router-link to="/" class="navbar-logo">
          Airline Satisfaction
        </router-link>
        <ul class="navbar-links">
          <li v-if="!isLoggedIn">
            <router-link to="/">Login</router-link>
          </li>
          <li v-if="!isLoggedIn">
            <router-link to="/register">Register</router-link>
          </li>
          <li v-if="isLoggedIn">
            <router-link to="/dashboard">Dashboard</router-link>
          </li>
          <li v-if="isLoggedIn">
            <router-link to="/prediction">History</router-link>
          </li>
          <li v-if="isLoggedIn">
            <button @click="handleLogout" class="logout-btn">Logout</button>
          </li>
        </ul>
      </div>
    </nav>
  </div>
</template>

<script>
import { useAuthStore } from "@/stores/store"; // Ensure correct store path
import { computed } from "vue";
import { useRouter } from "vue-router";

export default {
  name: "NavBar",
  setup() {
    const store = useAuthStore();
    const router = useRouter();

    const isLoggedIn = computed(() => store.isLoggedIn);

    // Logout function
    const handleLogout = () => {
      store.logout();
      router.push("/"); // Redirect to the login page after logout
    };

    return {
      isLoggedIn,
      handleLogout,
    };
  },
};
</script>

<style scoped>
.navbar {
  background-color: #fff;
  color: #333;
  padding: 1rem;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.navbar-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 1200px;
  margin: 0 auto;
}

.navbar-logo {
  font-size: 1.8rem;
  font-weight: bold;
  color: #42b983;
  text-decoration: none;
  text-transform: uppercase;
}

.navbar-links {
  list-style-type: none;
  display: flex;
  gap: 1.5rem;
  margin-bottom: 10px;
}

.navbar-links li {
  display: inline;
  font-family: "Trebuchet MS", "Lucida Sans Unicode", "Lucida Grande",
    "Lucida Sans", Arial, sans-serif;
  font-size: xx-large;
}

.navbar-links a {
  color: #333;
  text-decoration: none;
  font-size: 1.1rem;
}

.navbar-links a:hover {
  text-decoration: underline;
  color: #42b983;
}

.logout-btn {
  padding: 10px 20px;
  background-color: #ff6b6b;
  color: white;
  border: none;
  cursor: pointer;
  font-size: 1.1rem;
  font-family: "Trebuchet MS", "Lucida Sans Unicode", "Lucida Grande",
    "Lucida Sans", Arial, sans-serif;
  border-radius: 5px;
}

.logout-btn:hover {
  background-color: #e74c3c;
}
</style>
