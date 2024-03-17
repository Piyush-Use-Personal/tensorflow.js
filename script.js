

const urls = {
    model: 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json',
    metadata: 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json'
};
 
async function loadModel(url) {
    try {
        const model = await tf.loadLayersModel(urls.model);
        return model;
    } catch (err) {
        console.log(err);
    }
}
 
async function loadMetadata(url) {
    try {
        const metadataJson = await fetch(urls.metadata);
        const metadata = await metadataJson.json();
        return metadata;
    } catch (err) {
        console.log(err);
    }
}

function getSentimentScore(text, model, metadata) {
    const inputText = text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
    // Convert the words to a sequence of word indices.
    const sequence = inputText.map(word => {
        let wordIndex = metadata.word_index[word] + metadata.index_from;
        if (wordIndex > metadata.vocabulary_size) {
            wordIndex = -1;
        }
        return wordIndex;
    });
    // Perform truncation and padding.
    const paddedSequence = padSequences([sequence], metadata.max_len);
    const input = tf.tensor2d(paddedSequence, [1, metadata.max_len]);

    const predictOut = model.predict(input);
    const score = predictOut.dataSync()[0];
    predictOut.dispose();

    return score;
}

// Adjust padding logic to create a 2D array
function padSequences(sequences, maxlen) {
    return sequences.map(sequence => {
        if (sequence.length < maxlen) {
            return Array(maxlen - sequence.length).fill(0).concat(sequence);
        } else {
            return sequence.slice(sequence.length - maxlen, sequence.length);
        }
    });
}

async function classifyText() {
    let text = document.getElementById('textInput').value;
    document.getElementById('result').innerText = `loading`;

    const model = await loadModel()
    const metadata = await loadMetadata()
    const score = getSentimentScore(text, model, metadata);

    // Convert score to a rating out of 10
    const rating = (score * 10).toFixed(1);

    // Display result
    document.getElementById('result').innerText = `Sentiment: ${rating}/10`;
}
