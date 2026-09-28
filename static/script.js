// Two small helpers:
// 1. Clicking an example chip fills the input box.
// 2. While the server is working, the button shows "Finding jobs...".

document.querySelectorAll(".chip").forEach(function (chip) {
  chip.addEventListener("click", function () {
    document.getElementById("skills").value = chip.dataset.skills;
  });
});

document.getElementById("skills-form").addEventListener("submit", function () {
  var button = document.getElementById("submit-btn");
  button.disabled = true;
  button.textContent = "Finding jobs...";
});
