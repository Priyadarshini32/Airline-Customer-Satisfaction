<template>
  <v-app>
    <NavBar />
    <Router-view />
  </v-app>
</template>

<script setup>
import NavBar from "./views/NavBar.vue";
</script>

<style scoped>
/* Customize the background for v-app if needed */
.v-application {
  background-color: transparent; /* Or set any color you prefer */
}
</style>
