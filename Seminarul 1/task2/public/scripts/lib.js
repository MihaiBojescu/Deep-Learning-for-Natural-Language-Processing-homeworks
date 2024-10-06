export const sleep = (milliseconds) => new Promise(resolve => setTimeout(() => resolve(), milliseconds))

export const makeDeferredPromise = () => {
    let resolve = null
    let reject = null

    const promise = new Promise((innerResolve, innerReject) => {
        resolve = innerResolve
        reject = innerReject
    })

    return {
        promise,
        resolve,
        reject,
    }
}