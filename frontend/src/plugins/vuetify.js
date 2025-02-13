// src/plugins/vuetify.ts

// Styles
import "@mdi/font/css/materialdesignicons.css"; // Ensure you are using css-loader
import "vuetify/styles";

// Vuetify
import { createVuetify } from "vuetify";
import { aliases, mdi } from "vuetify/iconsets/mdi";

export default createVuetify({
  icons: {
    defaultSet: "mdi",
    aliases,
    sets: {
      mdi,
    },
  },
  theme: {
    defaultTheme: "light",
    themes: {
      light: {
        colors: {
          primary: "#3498db",
          secondary: "#2980b9",
          background: "#f5f5f5",
          surface: "#ffffff",
          error: "#e74c3c",
          text: "#333",
        },
      },
    },
  },
});
