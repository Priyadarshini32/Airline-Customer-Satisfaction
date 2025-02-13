<template>
  <div class="rating-container px-10">
    <span
      v-for="n in 5"
      :key="n"
      class="rating-star"
      :class="{ filled: n <= currentRating }"
      @click="updateRating(n)"
    >
      ★
    </span>
  </div>
</template>

<script setup>
import { ref, defineProps, defineEmits, watch } from "vue";

// Define props and emits
const props = defineProps({
  modelValue: {
    type: Number,
    required: true,
  },
});

const emit = defineEmits(["update:modelValue"]);

const currentRating = ref(props.modelValue);

// Watch for changes in the prop to keep the internal value in sync
watch(
  () => props.modelValue,
  (newValue) => {
    currentRating.value = newValue;
  }
);

const updateRating = (rating) => {
  currentRating.value = rating;
  // Emit the new rating value to the parent component
  emit("update:modelValue", rating);
};
</script>

<style scoped>
.rating-grid {
  display: flex;
  justify-content: space-between;
  gap: 20px;
}

.rating-column {
  flex: 1;
}

.rating-star {
  display: inline-block;
  font-size: 24px;
  color: #ddd;
  cursor: pointer;
  padding: 5px;
}

.rating-star.filled {
  color: #f39c12;
}
</style>
