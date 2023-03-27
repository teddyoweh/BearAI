class CosineSimilarity {
    constructor(text1, text2) {
        this.tokens1 = text1.toLowerCase().split(' ');
        this.tokens2 = text2.toLowerCase().split(' ');
        this.all_tokens = new Set([...this.tokens1, ...this.tokens2]);
        this.vector1 = Array.from(this.all_tokens, token => this.tokens1.filter(t => t === token).length);
        this.vector2 = Array.from(this.all_tokens, token => this.tokens2.filter(t => t === token).length);
    }
    
    get value() {
        const dot = this.vector1.reduce((acc, cur, i) => acc + cur * this.vector2[i], 0);
        const norm_x = Math.sqrt(this.vector1.reduce((acc, cur) => acc + cur ** 2, 0));
        const norm_y = Math.sqrt(this.vector2.reduce((acc, cur) => acc + cur ** 2, 0));
        return dot / (norm_x * norm_y);
    }
    
    get_similar_words(threshold) {
        const similar_words = [];
        for (const token of this.all_tokens) {
            const count1 = this.vector1[Array.from(this.all_tokens).indexOf(token)];
            const count2 = this.vector2[Array.from(this.all_tokens).indexOf(token)];
            if (count1 * count2 > 0 && (count1 * count2) / (Math.sqrt(count1 ** 2) * Math.sqrt(count2 ** 2)) >= threshold) {
                similar_words.push(token);
            }
        }
        return similar_words;
    }
}
