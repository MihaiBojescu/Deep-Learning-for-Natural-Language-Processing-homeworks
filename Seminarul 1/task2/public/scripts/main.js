import { makeDeferredPromise, sleep } from "./lib.js"

export const main = async () => {
    const promptsElement = document.querySelector("#prompts")
    const queryElement = document.querySelector("#query")
    const scoreElement = document.querySelector("#score")
    let score = 0

    const showWordAndScore = makeShowWordAndScore(promptsElement, queryElement, scoreElement)
    const queryUser = makeQueryUser(queryElement)
    const hideQueryAndScore = makeHideQueryAndScore(queryElement, scoreElement)
    const showSimilarity = makeShowSimilarity(promptsElement)
    const updateScore = makeUpdateScore(scoreElement)

    await intro(promptsElement)
    updateScore(score)

    while (true) {
        const word = await getWord()
        showWordAndScore(word)
        const userWord = await queryUser()
        hideQueryAndScore()
        const similarity = await querySimilarity(word, userWord)
        score += similarity
        await showSimilarity(word, userWord, similarity)
        updateScore(score)
    }
}

const intro = async (promptsElement) => {
    const sentence = "Welcome to word guesser . .. ..."


    for (const token of sentence.split(" ")) {
        promptsElement.innerHTML = `<p>${token}</p>`
        await sleep(500)
    }

    promptsElement.innerHTML = ""
}

const getWord = async () => {
    const request = await fetch("/api/words")
    const response = await request.json()

    return response.word
}

const makeShowWordAndScore = (promptsElement, queryElement, scoreElement) => (word) => {
    queryElement.classList.add("shown")
    scoreElement.classList.add("shown")

    promptsElement.innerHTML = `<p>Your word is "${word}". Write a similar word.</p>`

    queryElement.querySelector('input[name="query"]')?.focus()
}

const makeQueryUser = queryElement => async () => {
    const deferredPromise = makeDeferredPromise()
    const readWord = makeReadWord(deferredPromise)

    queryElement.addEventListener("submit", readWord)
    const query = await deferredPromise.promise
    queryElement.removeEventListener("submit", readWord)

    return query
}

const makeHideQueryAndScore = (queryElement, scoreElement) => () => {
    queryElement.classList.remove("shown")
    scoreElement.classList.remove("shown")
}

const makeReadWord = (promise) => async (event) => {
    event.preventDefault()
    const formData = new FormData(event.target)
    const formProps = Object.fromEntries(formData)

    event.target.reset()
    promise.resolve(formProps.query)
}

const querySimilarity = async (word, userWord) => {
    const request = await fetch("/api/guesses", {
        method: "POST",
        headers: {
            'Content-type': 'application/json'
        },
        body: JSON.stringify({
            word,
            userWord
        })
    })
    const response = await request.json()

    return response.similarity
}

const makeShowSimilarity = (promptsElement) => async (word, userWord, similarity) => {
    let output = ''

    if (similarity > 0.8) {
        output = 'great in fact!'
    } else if (similarity > 0.5) {
        output = 'just ok.'
    } else if (similarity > 0.3) {
        output = 'quite bad to be honest.'
    } else {
        output = 'in fact, awful!'
    }

    promptsElement.innerHTML = `<p>The word was "${word}".</p>`
    await sleep(1000)
    promptsElement.innerHTML += `<p>Your word was "${userWord}".</p>`
    await sleep(1000)
    promptsElement.innerHTML += `<p>Your guess is... ${output}</p>`
    await sleep(1000)
    promptsElement.innerHTML += `<p>(similarity is ${similarity.toPrecision(2)})</p>`
    await sleep(2000)
}

const makeUpdateScore = (scoreElement) => (score) => {
    scoreElement.textContent = `Score: ${score.toPrecision(2)}`
}