//JavaScript code needs to be written here and connected to index file.
// Select Button
const predictBtn = document.getElementById("predictBtn");

// Button Click Event
predictBtn.addEventListener("click", async () => {

    // Get textarea value
    const newsText = document.getElementById("newsInput").value;

    // Check if textarea is empty
    if (newsText.trim() === "") {
        alert("Please enter a news article or headline.");
        return;
    }

    // Flask API URL
    const API_URL = "http://127.0.0.1:5000/predict";

    try {

        // Send request to Flask backend
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                news: newsText
            })
        });

        // Convert response into JSON
        const data = await response.json();

        // Display results
        document.getElementById("classification").innerText =
            data.classification;

        document.getElementById("confidence").innerText =
            data.confidence_score;

        document.getElementById("reasoning").innerText =
            data.reasoning;

    } catch (error) {

        console.error("Error:", error);

        alert("Unable to connect to Flask API.");

    }

});