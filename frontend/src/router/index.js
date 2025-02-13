import { createRouter, createWebHistory } from "vue-router";
import Login from "../views/Login.vue";
import Register from "../views/Register.vue";
import Dashboard from "../views/Dashboard.vue";
import PredictionPage from "@/views/PredictionPage.vue";

const routes = [
  { path: "/", component: Login },
  { path: "/register", component: Register },
  { path: "/dashboard", component: Dashboard },
  { path: "/prediction", component: PredictionPage },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
