window.addEventListener("DOMContentLoaded", () => {
  // have to add an event listener for "esc" keypress for some reason
  let searchDialog = document.querySelector(".search-dialog");
  // Do nothing here if search is not enabled.
  if (!searchDialog) return;
  window.addEventListener("keydown", (evt) => {
    if (evt.key === "Escape" && searchDialog.open) {
      searchDialog.close();
    }
  });
});
