
var w = new Tensor(1, 784);
var b = 0.1;

function infer(x) {
    return x * w + b;
}

function train(x, y, lr) {
    x.transpose();
    // console.log(x.prod(w));
    // console.log(w);
    // console.log(w);
    var a = x.prod(w) + b;
    var y_ = activation(a);

    var loss = y_ - y;
    // console.log('a:' + a);
    // console.log('y_:' + y_);
    // console.log('y:'+ y);
    // console.log('loss:' + loss);

    var g = loss * (activation(a) * (1-activation(a)));
    // console.log('g:' + g);

    // console.log(x.mat[0][0]);
    var gW = []
    for (var i=0; i<x.mat.length; i++) {
        gW.push(g * x.mat[i][0])
    }
    // console.log(gW);
    var gB = g;
    // console.log(w.mat[683][0]);
    for (var i=0; i<gW.length; i++) {
        if (lr * gW[i] === 0) {
            continue;
        }
        // console.log('i:'+i);
        // console.log(lr * gW[i]);
        // console.log(w.mat[i][0]);
        
        w.mat[i][0] = w.mat[i][0] - (lr * gW[i]);
        // console.log(w.mat[i][0]);
    }
    // console.log(w.mat[683][0]);
    b = b - lr * gB;

    return loss;
}

function activation(x) {
    return 1.0 / (1.0 + Math.exp(-x));
}