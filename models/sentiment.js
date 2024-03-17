class Sentiment {
    constructor() {
        this.urls = {
            model: 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json',
            metadata: 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json'
        };
        this.model = null;
        this.metadata = null;
    }

    async loadModel() {
        try {
            this.model = await tf.loadLayersModel(this.urls.model);
        } catch (err) {
            console.log(err);
        }
    }

    async loadMetadata() {
        try {
            const metadataJson = await fetch(this.urls.metadata);
            this.metadata = await metadataJson.json();
        } catch (err) {
            console.log(err);
        }
    }

    getSentimentScore(text) {
        const inputText = text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
        // Convert the words to a sequence of word indices.
        const sequence = inputText.map(word => {
            let wordIndex = this.metadata.word_index[word] + this.metadata.index_from;
            if (wordIndex > this.metadata.vocabulary_size) {
                wordIndex = -1;
            }
            return wordIndex;
        });
        // Perform truncation and padding.
        const paddedSequence = this.padSequences([sequence], this.metadata.max_len);
        const input = tf.tensor2d(paddedSequence, [1, this.metadata.max_len]);

        const predictOut = this.model.predict(input);
        const score = predictOut.dataSync()[0];
        predictOut.dispose();

        return score;
    }

    // Adjust padding logic to create a 2D array
    padSequences(sequences, maxlen) {
        return sequences.map(sequence => {
            if (sequence.length < maxlen) {
                return Array(maxlen - sequence.length).fill(0).concat(sequence);
            } else {
                return sequence.slice(sequence.length - maxlen, sequence.length);
            }
        });
    }
}
export default Sentiment;
