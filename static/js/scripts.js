function enableMedia(answer) {
	if (answer.value == "video") {
		document.getElementById("video_form").classList.remove("d-none");
		document.getElementById("image_form").classList.add("d-none");
	} else if (answer.value == "image") {
		document.getElementById("video_form").classList.add("d-none");
		document.getElementById("image_form").classList.remove("d-none");
	} else {
		document.getElementById("video_form").classList.add("d-none");
		document.getElementById("image_form").classList.add("d-none");
	}
}
