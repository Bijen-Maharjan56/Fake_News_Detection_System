//Select elements from HTML
const predictionBtn = document.getElementById("predictBtn");
const newsInput = document.getElementById("newsInput");
const loader = document.getElementById("loader");
const resultContainer = document.getElementById("resultContainer");
const classificationEl = document.getElementById("classification");
const classificationBox = document.getElementById("classificationBox");
const confidenceEl = document.getElementById("confidence");
const reasoningEl = document.getElementById("reasoning");
const errorBox = document.getElementById("errorBox");
const errorText = document.getElementById("errorText");
//Storing all elements in variables at the top

//Button Click Event
predictionBtn.addEventListener("click", async () => {

    //Step 1: Read user input
    const newsText = newsInput.value.trim();
    //.value gets the text typed in the textarea
    //.trim() removes spaces at start and end

    //Step 2: Validate
    if (newsText === ""){
        alert("Please enter a news article or headline.");
        return; //stops the function here
    }

    //Step 3: Reset UI before new prediction
    resultContainer.style.display = "none";
    errorBox.style.display = "none";

    classificationBox.className = "result-box";

    //Step 4: Show loader, disable button
    loader.style.display = "block";
    predictionBtn.disabled = true;
    predictionBtn.textContent = "...";

    try{
        //Step 5: Send text to Flask API
        const response = await fetch ("/predict", {
            method: "POST", //POST because we are sending data (news text)

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                news: newsText
                //Sends {"news": "the article text here"}
                //model.py will read this with incoming.get('news','')
            })
        });
        //Step 6: Read Flask's JSON response
        const data = await response.json();

        //Step 7: Handle error sent from Flask
        if(!response.ok || data.error){
            showError(data.error || "Something went wrong.");
            return;
        }
        //Step 8: Fill Classification Result Box
        if (data.prediction === "FAKE"){
            classificationEl.textContent = "FAKE NEWS";
            classificationBox.classList.add("fake");

        }else{
            classificationEl.textContent = "REAL NEWS";
            classificationBox.classList.add("real");
        }
        //Step 9: Fill Confidence Score box
        confidenceEl.textContent = data.confidence + "%";

        //Step 10: Fill Reasoning box
        reasoningEl.innerHTML = "";
        //Clear previous word list first

        data.top_words.forEach(item => {
            const li = document.createElement("li");
            const direction = item.score > 0
                ? '<span class="direction">-> supports REAL </span>'
                : '<span class="direction">-> supports FAKE</span>';

                li.innerHTML = `<b>${item.word}</b>: ${item.score} ${direction}`;
                reasoningEl.appendChild(li);
        });

        //Step 11: Show result section
        resultContainer.style.display = "flex";
    }catch (error) {
        //Runs if Flask server is completely unreachable
        showError("Cannot connect to server. Make sure app.py is running.")
        console.error("Fetch error:", error);
    }finally {
        loader.style.display = "none";
        predictionBtn.disabled = false;
        predictionBtn.textContent = "Predict";
    }
});

function showError(message){
    errorText.textContent = "⚠️" + message;
    errorBox.style.display = "block";
    loader.style.display = "none";
    predictionBtn.disabled = false;
    predictionBtn.textContent = "Predict";
}