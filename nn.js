var w = 0.1
var b = 4.0

function infer(x) {
    return x * w + b;
}

function train(x, y, lr) {
    var y_ = []
    for (var i=0; i<x.length; i++) {
        y__ = x[i] * w + b;
        y_.push(y__)
    }
    // console.log(y_);

    var loss = []
    for (var i=0; i<x.length; i++) {
        loss__ = y_[i] - y[i];
        loss.push(loss__);
    }
    // console.log(loss);

    var gW = []
    var gB = []
    for (var i=0; i<x.length; i++) {
        gW_ = loss[i] * x[i];
        gB_ = loss[i];
        gW.push(gW_)
        gB.push(gB_)
    }
    // console.log(gW);

    for (var i=0; i<x.length; i++) {
        w = w - (lr * gW[i]);
    }

    for (var i=0; i<x.length; i++) {
        b = b - (lr * gB[i]);
    }

    return loss[x.length-1];
}

