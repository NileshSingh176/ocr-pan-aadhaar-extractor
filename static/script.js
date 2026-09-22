document.getElementById("imageInput").addEventListener("change", function(event) {
    let file = event.target.files[0];

    if (file) {
        let reader = new FileReader();

        reader.onload = function(e) {
            let img = document.createElement("img");
            img.src = e.target.result;
            img.width = 200;

            let preview = document.getElementById("previewBox");
            preview.innerHTML = "";
            preview.appendChild(img);
        };

        reader.readAsDataURL(file);
    }
});