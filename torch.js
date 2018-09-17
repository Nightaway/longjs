"use strict";

const ADD = 1;
const MUL = 2;
const TRANSPOSE = 3;
const POW = 4;
const SUB = 5;
const DIV = 6;
const MINUS = 7;
const MAX = 8;

function Function(head) {
    this.head = head;
    this.next = head.backward1();
    this.grad = null;
    this.class = 'function';
    this.name = "";
}

Function.prototype.backward = function () {
    for (var i=0; i<this.next.length; i++) {
        this.choose(this.next[i]);
        var d = this.head.derive();
        this.unchoose();
 
        var grad = null;
        if (arguments.length === 1) {
            grad = arguments[0].mul(d);
        } else {
            grad = this.head.mul(d);
        }
        // console.log(this.name);
        // console.log(this.head);
        // console.log(arguments[0]);
        // console.log(d);
        // console.log(this.next[i].name);
        // console.log(grad);
        this.next[i].grad = grad;
        if (this.next[i].func !== null) {
            this.next[i].func.backward(grad);
        }
    }
}

Function.prototype.choose = function (v) {
    for (var i=0; i<this.next.length; i++) {
        if (this.next[i] !== v) {
            this.next[i].constant = true;
        }
    }
}

Function.prototype.unchoose = function () {
    for (var i=0; i<this.next.length; i++) {
        this.next[i].constant = false;
    }
}

Function.prototype.backward1 = function () {
    return this.head.backward1();
}

function Tensor() {
    var args = arguments[0];
    this.rows = 0;
    this.cols = 0;
    this.mat = [];
    this.grad = null;
    this.nodes = [];
    this.type = 2;
    this.op = 0;
    this.parent = null;
    this.constant = false;
    this.class = 'tensor';
    this.func = null;
    this.name = "";
    
    if (args.length === 1) {
        if (Array.isArray(args[0])) {
            this.rows = args.length;
            this.cols = args[0].length;
            this.mat.push(args[0]);
        } else if (typeof args[0] == "object") {
            if (args[0].class == "tensor") {
                this.rows = args[0].rows;
                this.cols = args[0].cols;
                this.mat = args[0].mat;
                this.grad = args[0].grad;
                this.nodes = args[0].nodes;
                this.type = args[0].type;
                this.op = args[0].op;
                this.parent = args[0].parent;
                this.constant = args[0].constant;
            } else if (args[0].class == "function") {
                this.rows = args[0].head.rows;
                this.cols = args[0].head.cols;
                this.mat = args[0].head.mat;
                this.func = args[0];
            } 
        } else {
            this.mat = args[0];
        }
    } else if (args.length === 2) {
        this.rows = args[0];
        this.cols = args[1];
        for (var i=0; i<this.rows; i++) {
            for (var j=0; j<this.cols; j++) {
                this.mat.push([0.00005]);
            }
        }
    } else if (args.length === 3) {
        this.rows = args[0];
        this.cols = args[1];
        for (var i=0; i<this.rows; i++) {
            for (var j=0; j<this.cols; j++) {
                this.mat.push([args[2]]);
            }
        }
    }
}

Tensor.prototype.max = function(x) {
    let l = torch.tensor(0);
    let r = torch.tensor(x);

    let z = torch.tensor();
    z.nodes.push(l);
    z.nodes[0].parent = this;
    z.nodes.push(r);
    z.nodes[1].parent = this;
    z.type = 1;
    z.op = MAX;
    return z;
}

Tensor.prototype.transpose = function () {
    if (this.rows == 0 && this.cols == 0) {
        return this;
    }

    var z = torch.tensor();
    for (var i=0; i<this.cols; i++) {
        var rows = [];
        for (var j=0; j<this.rows; j++) {
            rows.push(this.mat[j][i]);
        }
        z.mat.push(rows);
    }
    var temp = this.rows;
    z.rows = this.cols;
    z.cols = temp;
    
    z.nodes.push(this);
    z.nodes[0].parent = this;
    z.type = 1;
    z.op = TRANSPOSE;
    return z;
}

Tensor.prototype.transpose_ = function () {
    if (this.rows == 0 && this.cols == 0) {
        return this;
    }

    var z = torch.tensor();
    for (var i=0; i<this.cols; i++) {
        var rows = [];
        for (var j=0; j<this.rows; j++) {
            rows.push(this.mat[j][i]);
        }
        z.mat.push(rows);
    }
    var temp = this.rows;
    z.rows = this.cols;
    z.cols = temp;

    return z;
}

Tensor.prototype.mul = function (y) {
    var z = torch.tensor();
    if (typeof y == "number") {
        z.rows = this.rows;
        z.cols = this.cols;

        if (this.rows == 0 && this.cols == 0) {
            for (var i=0; i<this.rows; i++) {
                var row = [];
                for (var j=0; j<this.cols; j++) {
                    row.push(this.mat * y);
                }
                z.mat.push(row);
            }
        } else {
            for (var i=0; i<this.rows; i++) {
                var row = [];
                for (var j=0; j<this.cols; j++) {
                    row.push(this.mat[i][j] * y);
                }
                z.mat.push(row);
            }
        }
    } else {
        if (this.rows == 0 && this.cols == 0) {
            if (this.mat == 0) {
                z.rows = 0;
                z.cols = 0;
                z.mat = 0;
            } else if (y.rows == 0 && y.cols == 0) {
                z.rows = y.rows;
                z.cols = y.cols;
                z.mat = this.mat * y.mat;
            } else {
                z.rows = y.rows;
                z.cols = y.cols;
                for (var i=0; i<y.rows; i++) {
                    var row = [];
                    for (var j=0; j<y.cols; j++) {
                        row.push(this.mat * y.mat[i][j]);
                    }
                    z.mat.push(row);
                }
            }            
        } else if (this.rows == 1 && this.cols == 1) {
            if (this.mat[0] == 0) {
                z.rows = 0;
                z.cols = 0;
                z.mat = 0;
            } if (y.rows == 0 && y.cols == 0) {
                z.rows = y.rows;
                z.cols = y.cols;
                z.mat = this.mat[0][0] * y.mat;
            } else {
                z.rows = y.rows;
                z.cols = y.cols;
                z.mat = [];
                for (var i=0; i<y.rows; i++) {
                    var row = [];
                    for (var j=0; j<y.cols; j++) {
                        row.push(this.mat[0][0] * y.mat[i][j]);
                    }
                    z.mat.push(row);
                }
            }
        } else {
            if (y.cols == 0 && y.rows == 0) {
                z.rows = this.rows;
                z.cols = this.cols;
                for (var i=0; i<this.rows; i++) {
                    var row = [];
                    for (var j=0; j<this.cols; j++) {
                        row.push(this.mat[i][j] * y.mat);
                    }
                    z.mat.push(row);
                }
            } else {
                z.rows = this.rows;
                z.cols = y.cols;

                for (var i=0; i<this.rows; i++) {
                    var cols = [];
                    for (var j=0;j<y.cols; j++) {
                        var col = [];
                        for (var k=0; k<y.rows; k++) {
                            col.push(y.mat[k][j]);
                        }
                        cols.push(col);
                    }

                    var row = [];
                    for (var j=0; j<cols.length; j++) {
                        var sum = 0;
                        for (var k=0; k<cols[j].length; k++) {
                            sum += this.mat[i][k] * cols[j][k];
                        }
                        row.push(sum);
                    }
                    z.mat.push(row);
                }
            }
        }
    }
    z.nodes.push(this);
    z.nodes[0].parent = this;
    z.nodes.push(y);
    if (typeof y !== "number") {
        z.nodes[1].parent = this;
    }
    z.type = 1;
    z.op = MUL;
    return z;
}

Tensor.prototype.sub = function (y) {
    var z = torch.tensor();
    z.rows = y.rows;
    z.cols = y.cols;
    if (y.rows == 0 && y.cols == 0) {  
        if (this.rows == 0 && this.cols == 0) {
            z.mat = this.mat - y.mat;
        } else {
            z.rows = this.rows;
            z.cols = this.cols;
            for (var i=0; i<z.rows; i++) {
                var row = [];
                for (var j=0; j<z.cols; j++) {
                    row.push(this.mat[i][j] - y.mat);
                }
                z.mat.push(row);
            }
        }
        
    } else {
        if (this.rows == 0 && this.cols == 0) {
            for (var i=0; i<z.rows; i++) {
                var row = [];
                for (var j=0; j<z.cols; j++) {
                    row.push(this.mat - y.mat[i][j]);
                }
                z.mat.push(row);
            }
        } else {
            for (var i=0; i<z.rows; i++) {
                var row = [];
                for (var j=0; j<z.cols; j++) {
                    row.push(this.mat[i][j] - y.mat[i][j]);
                }
                z.mat.push(row);
            }
        }
    }

    z.nodes.push(this);
    z.nodes[0].parent = this;
    z.nodes.push(y);
    if (typeof y !== "number") {
        z.nodes[1].parent = this;
    }
    z.type = 1;
    z.op = SUB;
    return z;
}

Tensor.prototype.div = function (y) {
    var z = torch.tensor();
    z.rows = y.rows;
    z.cols = y.cols;

    if (this.rows == 0 && this.cols == 0) {
        if (y.rows == 0 && y.cols == 0) {
            z.mat = this.mat / y.mat;
        } else {
            for (var i=0; i<z.rows; i++) {
                var row = [];
                for (var j=0; j<z.cols; j++) {
                    row.push(this.mat / y.mat[i][j]);
                }
                z.mat.push(row);
            }
        }
    } else {
        for (var i=0; i<z.rows; i++) {
            var row = [];
            for (var j=0; j<z.cols; j++) {
                row.push(this.mat[i][j] / y.mat[i][j]);
            }
            z.mat.push(row);
        }
    }
    
    z.nodes.push(this);
    z.nodes[0].parent = this;
    z.nodes.push(y);
    z.nodes[1].parent = this;
    z.type = 1;
    z.op = DIV;
    return z;
}

Tensor.prototype.pow = function (y) {
    var z = torch.tensor();
    z.rows = this.rows;
    z.cols = this.cols;

    if (z.rows == 0 && z.cols == 0) {
        if (typeof y == "number") {
            z.mat = Math.pow(this.mat, y);
        } else if (y.rows == 0 && y.cols == 0) {
            z.mat = Math.pow(this.mat, y.mat);
        } else {
            z.rows = y.rows;
            z.cols = y.cols;
            for (var i=0; i<z.rows; i++) {
                var row = [];
                for (var j=0; j<z.cols; j++) {
                    row.push(Math.pow(this.mat, y.mat[i][j]));
                }
                z.mat.push(row);
            }
        }
    } else {
        for (var i=0; i<this.rows; i++) {
            var row = [];
            for (var j=0; j<this.cols; j++) {
                row.push(Math.pow(this.mat[i][j], y));
            }
            z.mat.push(row);
        }
    }

    z.nodes.push(this);
    z.nodes[0].parent = this;
    z.nodes.push(y);
    if (typeof y !== "number") {
        z.nodes[1].parent = this;
    }
    z.type = 1;
    z.op = POW;
    return z;
}

Tensor.prototype.mul1 = function (y) {
    var z = torch.tensor();
    if (typeof y == "number") {
        z.rows = this.rows;
        z.cols = this.cols;
        for (var i=0; i<z.rows; i++) {
            var row = [];
            for (var j=0; j<z.cols; j++) {
                row.push(this.mat[i][j] * y);
            }
            z.mat.push(row);
        }
    } else {
        z.rows = y.rows;
        z.cols = y.cols;
        for (var i=0; i<z.rows; i++) {
            var row = [];
            for (var j=0; j<z.cols; j++) {
                row.push(this.mat[0][0] * y.mat[i][j]);
            }
            z.mat.push(row);
        }
    }
    return z;
}

Tensor.prototype.mul2 = function (y) {
    var z = torch.tensor();
    z.rows = y.rows;
    z.cols = y.cols;
    for (var i=0; i<z.rows; i++) {
        var row = [];
        for (var j=0; j<z.cols; j++) {
            row.push(this.mat[i][j] * y.mat[i][j]);
        }
        z.mat.push(row);
    }
    return z;
}

Tensor.prototype.add = function (y) {
    var z = torch.tensor();
    z.rows = this.rows;
    z.cols = this.cols;

    if (y.rows == 0 && y.cols == 0) {
        if (this.rows == 0 && this.cols == 0) {
            z.mat = this.mat + y.mat;
        } else {
            for (var i=0; i<z.rows; i++) {
                var row = [];
                for (var j=0; j<z.cols; j++) {
                    row.push(this.mat[i][j] + y.mat);
                }
                z.mat.push(row);
            }
        }
    } else {
        if (this.rows == 0 && this.cols == 0) {
            z.rows = y.rows;
            z.cols = y.cols;
            for (var i=0; i<z.rows; i++) {
                var row = [];
                for (var j=0; j<z.cols; j++) {
                    row.push(this.mat + y.mat[i][j]);
                }
                z.mat.push(row);
            }
        } else {
            for (var i=0; i<z.rows; i++) {
                var row = [];
                for (var j=0; j<z.cols; j++) {
                    row.push(this.mat[i][j] + y.mat[i][j]);
                }
                z.mat.push(row);
            }
        }
    }

    z.nodes.push(this);
    z.nodes[0].parent = this;
    z.nodes.push(y);
    if (typeof y !== "number") {
        z.nodes[1].parent = this;
    }
    z.type = 1;
    z.op = ADD;
    return z;
}

Tensor.prototype.sub_ = function (y) {
    if (this.rows == 0 && this.cols == 0) {
        if (y.rows == 0 && y.cols == 0) {
            this.mat -= y.mat;
        }
    } else {
        if (y.rows == 0 && y.cols == 0) {
            for (var i=0; i<this.rows; i++) {
                for (var j=0; j<this.cols; j++) {
                    this.mat[i][j] -= y.mat;
                }
            }
        } else {
            for (var i=0; i<this.rows; i++) {
                for (var j=0; j<this.cols; j++) {
                    this.mat[i][j] -= y.mat[i][j];
                }
            }
        }
    }
}

Tensor.prototype.minus = function () {
    var z = torch.tensor();
    z.rows = this.rows;
    z.cols = this.cols;

    if (this.rows == 0 && this.cols == 0) {
        z.mat = -this.mat;
    } else {
        for (var i=0; i<z.rows; i++) {
            var rows = [];
            for (var j=0; j<z.cols; j++) {
               rows.push(-this.mat[i][j]);
            }
            z.mat.push(rows);
        }
    }

    z.nodes.push(this);
    z.nodes[0].parent = this;
    z.type = 1;
    z.op = MINUS;
    return z;
}

Tensor.prototype.derive = function(v) {
    if (this.constant) {
        return 0;
    }
    if (this.func !== null) {
        return 1;
    }

    switch (this.type) {
        case 1:
            switch(this.op) {
                case ADD:
                var l = this.nodes[0].derive();
                var r = this.nodes[1].derive();
                // console.log(l);
                // console.log(r);
                if (typeof l === "number") {
                    l = torch.tensor(l);
                }
                if (typeof r === "number") {
                    r = torch.tensor(r);
                }
                // console.log(l.add(r));
                return l.add(r);

                case SUB:
                var l = this.nodes[0].derive();
                var r = this.nodes[1].derive();
                
                // console.log(l);
                // console.log(r);
                if (typeof r === "number") {
                    r = torch.tensor(r);
                }
                if (typeof l === "number") {
                    l = torch.tensor(l);
                }
                // console.log(l);
                // console.log(r);
                // console.log(l.sub(r));
                return l.sub(r);

                case MUL:
                var l = this.nodes[0].derive();
                var r = this.nodes[1].derive();
                if (typeof r === "number") {
                    r = torch.tensor(r);
                }
                if (typeof l === "number") {
                    l = torch.tensor(l);
                }

                // if (!this.nodes[0].constant) {
                //     return this.nodes[1];
                // } else {
                //     return this.nodes[0];
                // }
                // console.log(this.nodes[0].constant);
                // console.log(this.nodes[1].constant);
                return (l.mul(this.nodes[1])).add((r.mul(this.nodes[0])));

                case TRANSPOSE:
                var l = this.nodes[0].derive();
                return l;

                case POW:
                var l = this.nodes[0].derive();
                var r = this.nodes[1].derive();
                // console.log(l);
                // console.log(r * Math.pow(this.nodes[0].mat, this.nodes[1].mat));
                if (this.nodes[0].constant) {
                    if (this.nodes[1].rows == 0 && this.nodes[1].cols == 0) {
                        return r * Math.pow(this.nodes[0].mat, this.nodes[1].mat);
                    } else {
                        // console.log(this.nodes[0].pow(this.nodes[1]).mul(r));
                        return this.nodes[0].pow(this.nodes[1]).mul(r);
                    }
                }
                break;

                case DIV:
                var l = this.nodes[0].derive();
                var r = this.nodes[1].derive();
                // console.log(r);
                // console.log(this.nodes[0].mat);
                // console.log((l * this.nodes[1].mat - this.nodes[0].mat * r));
                if (this.nodes[1].rows == 0 && this.nodes[1].cols == 0) {
                    // console.log((l * this.nodes[1].mat - this.nodes[0].mat * r) / Math.pow(this.nodes[1].mat, 2));
                    return (l * this.nodes[1].mat - this.nodes[0].mat* r.mat) / Math.pow(this.nodes[1].mat, 2);
                } else {
                    // var lhr = this.nodes[1].mul(l);
                    // var rhr = this.nodes[0].mul(r);
                    // var thrd = this.nodes[1].pow(2);
                    // console.log('1111');
                    // console.log(thrd);
                    // console.log(rhr);
                    // console.log(lhr.sub(rhr).div(thrd));
                    // console.log((this.nodes[1].mul(l).sub(this.nodes[0].mul(r))).div(this.nodes[1].pow(2)));
                    return (this.nodes[1].mul(l).sub(this.nodes[0].mul(r))).div(this.nodes[1].pow(2));
                }
            
                case MINUS:
                var l = this.nodes[0].derive();
                // console.log(this.nodes);
                // console.log(r);
                return -1;
            }
        break;

        case 2:
            if (this.constant) {
                return 0;
            } else if (this.function !== null) {
                return 1;
            }
        break;
    }
}

Tensor.prototype.backward1 = function() {
    let vars = [];
    for (var i=0; i<this.nodes.length; i++) {
        if (this.nodes[i].constant === false && this.nodes[i].op === 0) {
            vars.push(this.nodes[i]);
        }
        let next = this.nodes[i].backward1();
        if (next.length > 0) {
            vars = vars.concat(next);
        }
    }
    return vars;
}

class torch {
    static tensor() {
		return new Tensor(arguments);
    }
    
    static const() {
        var tensor = new Tensor(arguments);
        tensor.constant = true;
		return tensor;
    }
    
    static function(head) {
        return new Function(head);
    }
}

class F {
    static sigmoid(x) {
        return torch.function(torch.const(1).div(torch.const(1).add(torch.const(Math.E).pow(torch.tensor(x).minus()))));
    }

    static relu(x) {
        return torch.function(torch.max(x));
    }

    static softmax(x) {
        let sum = torch.tensor();
        for (let i=0; i<x.length; x++) {
            sum.add(torch.const(Math.E).pow(torch.tensor(x[i])));
        }
    }
}

class nn {
    static Linear(input_size, output_size) {
        return function(x) {
            let w = torch.tensor(input_size, output_size);
            let b = torch.tensor();
            return torch.function(x.mul(w).add(b));
        }
    }

    static ReLU() {

    }

    static Sequentail() {
        
    }
}