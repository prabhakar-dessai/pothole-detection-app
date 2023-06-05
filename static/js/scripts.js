//Scrolling to DEMO Section
function scrollToDemo() {
	const targetDiv = document.getElementById("demoDiv");
	targetDiv.scrollIntoView({ behavior: "smooth" });
}

//Scrolling to About us Section
function scrollToAbout() {
	const targetDiv = document.getElementById("aboutUs");
	targetDiv.scrollIntoView({ behavior: "smooth" });
}

//Handles the Selection of Image or a Video
$(document).ready(function () {
	$(".selectpicker").selectpicker();

	//Function to clear the image and its corresponding results
	function clearImageAndResults() {
		var processedImageContainer = document.getElementById(
			"processed-image-container"
		);
		var inputImage = document.getElementById("input-image");
		processedImageContainer.innerHTML = "";
		inputImage.src = "";
	}

	// Function to handle section visibility based on select box value
	function handleSectionVisibility() {
		var selectBox = document.querySelector(".selectpicker");
		var imageSection = document.getElementById("image-section");
		var videoSection = document.getElementById("video-section");

		selectBox.addEventListener("change", function () {
			var selectedOption = this.value;
			if (selectedOption === "Images") {
				imageSection.style.display = "block";
				videoSection.style.display = "none";
				clearImageAndResults();
			} else if (selectedOption === "Videos") {
				imageSection.style.display = "none";
				videoSection.style.display = "block";
				clearImageAndResults();
			}
		});
	}

	var imageSection = document.getElementById("image-section");
	var videoSection = document.getElementById("video-section");

	// Hide the sections initially
	imageSection.style.display = "none";
	videoSection.style.display = "none";
	handleSectionVisibility();
});

// Function to handle the choose an image and preview option
document.addEventListener("DOMContentLoaded", function () {
	const imageFileInput = document.getElementById("img");
	const videoFileInput = document.getElementById("video");
	const imageFileNameLabel = document.getElementById("image-file-name");
	const videoFileNameLabel = document.getElementById("video-file-name");
	const imageCloseFile = document.getElementById("image-close-file");
	const videoCloseFile = document.getElementById("video-close-file");
	const imageFileInfo = document.getElementById("image-file-info");
	const videoFileInfo = document.getElementById("video-file-info");
	const imagePreviewImage = document.getElementById("image-preview-image");
	const videoPreview = document.getElementById("video-preview-video");
	const imageUploadText = document.getElementById("image-upload-text");
	const videoUploadText = document.getElementById("video-upload-text");
	const uploadButton = document.querySelectorAll(".upload-button");
	const progressBar = document.querySelectorAll(".progress");

	function handleImageSelection() {
		const file = imageFileInput.files[0];
		if (file) {
			const reader = new FileReader();

			reader.onload = function (event) {
				const imageUrl = event.target.result;
				imagePreviewImage.src = imageUrl;
				imageFileInfo.style.display = "flex";
				imageFileNameLabel.textContent = file.name;
				imagePreviewImage.style.display = "block";
				imageUploadText.style.display = "none";
			};

			reader.readAsDataURL(file);
		} else {
			clearImageSelection();
		}
	}

	function handleVideoSelection() {
		const file = videoFileInput.files[0];
		if (file) {
			const videoUrl = URL.createObjectURL(file);

			videoPreview.src = videoUrl;
			videoFileInfo.style.display = "flex";
			videoFileNameLabel.textContent = file.name;
			videoPreview.style.display = "block";
			videoUploadText.style.display = "none";

			//To adjust the dimensions to match the container
			videoPreview.style.width = "100%";
			videoPreview.style.height = "100%";
		} else {
			clearVideoSelection();
		}
	}

	function clearImageSelection() {
		imageFileInput.value = "";
		imagePreviewImage.src = "";
		imageFileInfo.style.display = "none";
		imageFileNameLabel.textContent = "";
		imagePreviewImage.style.display = "none";
		imageUploadText.style.display = "block";
	}

	function clearVideoSelection() {
		videoFileInput.value = "";
		videoPreview.src = "";
		videoFileInfo.style.display = "none";
		videoFileNameLabel.textContent = "";
		videoPreview.style.display = "none";
		videoUploadText.style.display = "block";
	}

	function handleProgress() {
		progressBar.forEach((bar) => {
			bar.style.display = "block";
		});
	}

	imageFileInput.addEventListener("change", handleImageSelection);
	videoFileInput.addEventListener("change", handleVideoSelection);
	imageCloseFile.addEventListener("click", clearImageSelection);
	videoCloseFile.addEventListener("click", clearVideoSelection);
	uploadButton.forEach((btn) => {
		btn.addEventListener("click", handleProgress);
	});

	// Hide the initial image and video previews
	progressBar.forEach((bar) => {
		bar.style.display = "none";
	});
	imageFileInfo.style.display = "none";
	videoFileInfo.style.display = "none";
	imagePreviewImage.style.display = "none";
	videoPreview.style.display = "none";
});
