const fs = require('fs').promises;
const path = require('path');

async function replaceMarker() {
    let template = await fs.readFile("./template.html", 'utf8');
    let replacement = await fs.readFile("./words.html", 'utf8');
    let result = template.replace('[[REPLACE_WITH_WORDS_HTML]]', replacement);
    await fs.writeFile("./index.html", result, 'utf8');
}

replaceMarker();
