"use strict";

const ADD = 'add';
const MUL = 'mul';
const TRANSPOSE = 'transpose';
const POW = 'pow';
const SUB = 'sub';
const DIV = 'div';
const MINUS = 'minus';
const MAX = 'max';
const REDUCE_SUM = 'reduce_sum';

function Function(head, name) {
    this.head = head;
    this.vars = head.findAllVaribles();
    this.grad = null;
    this.class = 'function';
    this.name = name;
}

Function.prototype.backward = function () {
    for (var i=0; i<this.vars.length; i++) {
        this.choose(this.vars[i]);
        var d = this.head.derive(arguments[0]);
        this.unchoose();
 
        var grad = d;
        if (d.isgrad == false) {
            if (arguments.length === 1) {
                // console.log(1);
                // console.log(this.name);
                
                if (d.rows > 1 || d.cols > 1) {
                    if (arguments[0].cols != d.rows) {
                        d = d.transpose();
                    }

                    if (arguments[0].cols != d.rows) {
                        arguments[0] = arguments[0].transpose();
                        if (arguments[0].cols != d.rows) { 
                            d = d.transpose();
                        }
                    }
                }
                // console.log(arguments[0]);
                // console.log(d);
                grad = arguments[0].mul(d);
                // console.log(grad);
            } else {
                // console.log(2);
                // console.log(this.name);
                // console.log(this.head);
                if (d.rows > 1 || d.cols > 1) {
                    if (this.head.cols != d.rows) {
                        d = d.transpose();
                    }
                }
                // console.log(d);
                grad = this.head.mul(d);
                // console.log(grad);
            }
        } else {
            grad = d;
            // console.log(3);
            // console.log(this.name);
            // console.log(grad);
        }

        this.vars[i].grad = grad;
        if (this.vars[i].func !== null) {
            this.vars[i].func.backward(this.vars[i].grad);
        }
    }
}

Function.prototype.choose = function(v) {
    for (var i=0; i<this.vars.length; i++) {
        if (this.vars[i] !== v) {
            this.vars[i].constant = true;
        }
    }
}

Function.prototype.unchoose = function () {
    for (var i=0; i<this.vars.length; i++) {
        this.vars[i].constant = false;
    }
}

Function.prototype.findAllVaribles = function () {
    return this.head.findAllVaribles();
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
    this.isgrad = false;
    this.constant = false;
    this.class = 'tensor';
    this.func = null;
    this.name = "";
    this.require_grad = false;
    
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
                // this.parent = args[0].parent;
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
            let row = [];
            for (var j=0; j<this.cols; j++) {
                row.push(0.00005);
            }
            this.mat.push(row);
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

Tensor.prototype.max = function() {
    let l = torch.tensor(0);
    let r = torch.tensor(this);
    let z = torch.tensor();

    if (typeof(this) === "number") {

    } else {
        if (this.rows === 0 && this.cols === 0) {

        } else {
            z.rows = this.rows;
            z.cols = this.cols;
            for (let i=0; i<this.rows; i++) {
                let row = [];
                for (let j=0; j<this.cols; j++) {
                    row.push(Math.max(0, this.mat[i][j]));
                }
                z.mat.push(row);
            }
        }
    }

    z.nodes.push(l);
    // z.nodes[0].parent = this;
    z.nodes.push(this);
    // z.nodes[1].parent = this;
    z.type = 1;
    z.op = MAX;
    // console.log(this);
    return z;
}

Tensor.prototype.reduce_sum = function () {
    var z = torch.const();
    z.rows = 1;
    z.cols = 1;
    var sum = 0;
    for (var i=0; i<this.rows; i++) {
        for (var j=0; j<this.cols; j++) {
            sum += this.mat[i][j];
        }
    }
    var v = torch.const();
    v.rows = this.rows;
    v.cols = this.cols;
    v.mat = this.mat;
    z.nodes.push(v);
    z.mat.push([sum]);
    z.type = 1;
    z.op = REDUCE_SUM;
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
    // z.nodes[0].parent = this;
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

Tensor.prototype.multiply = function (y) {
    var z = torch.tensor();
    z.rows = this.rows;
    z.cols = this.cols;
    for (var i=0; i<this.rows; i++) {
        var row = [];
        for (var j=0;j<this.cols; j++) {
            row.push(this.mat[i][j]*y.mat[i][j]);
        }
        z.mat.push(row);
    }
    return z;
}

Tensor.prototype.mul = function (y) {
    if (this.rows == y.rows && this.cols == y.cols) {
        return this.multiply(y);
    }

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
    // z.nodes[0].parent = this;
    z.nodes.push(y);
    // if (typeof y !== "number") {
    //     z.nodes[1].parent = this;
    // }
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
        } else if (this.rows == 1 && this.cols == 1) {
            z.rows = 1;
            z.cols = 1;
            z.mat.push([this.mat[0][0] - y.mat]);
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
            if (y.rows == 1 && y.cols == 1) {
                var row = [];
                row.push(this.mat - y.mat[0][0]);
                z.mat.push(row);
            } else {
                for (var i=0; i<z.rows; i++) {
                    var row = [];
                    for (var j=0; j<z.cols; j++) {
                        row.push(this.mat - y.mat[i][j]);
                    }
                    z.mat.push(row);
                }
            }
        } else if (this.rows == 1 && this.cols == 1) {
            if (y.rows == 0 && y.cols == 0) {
                z.mat = this.mat[0][0] - y.mat;
            } else if (y.rows == 1 && y.cols == 1) {
                z.mat.push([this.mat[0][0] - y.mat[0][0]]);
            }
        } 
        else {
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
    // z.nodes[0].parent = this;
    z.nodes.push(y);
    // if (typeof y !== "number") {
    //     z.nodes[1].parent = this;
    // }
    z.type = 1;
    z.op = SUB;
    return z;
}

Tensor.prototype.div = function (y) {
    var z = torch.tensor();
    z.rows = y.rows;
    z.cols = y.cols;
    // console.log(this);
    // console.log(y);
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
        if (y.rows == 1 && y.cols == 1) {
            z.rows = this.rows;
            z.cols = this.cols;
            for (var i=0; i<z.rows; i++) {
                var row = [];
                for (var j=0; j<z.cols; j++) {
                    row.push(this.mat[i][j] / y.mat[0][0]);
                }
                z.mat.push(row);
            }
        } else if (y.rows == 0 && y.cols == 0) {
            z.rows = this.rows;
            z.cols = this.cols;
            for (var i=0; i<z.rows; i++) {
                var row = [];
                for (var j=0; j<z.cols; j++) {
                    row.push(this.mat[i][j] / y.mat);
                }
                z.mat.push(row);
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
    }
    
    z.nodes.push(this);
    // z.nodes[0].parent = this;
    z.nodes.push(y);
    // z.nodes[1].parent = this;
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
    // z.nodes[0].parent = this;
    z.nodes.push(y);
    // if (typeof y !== "number") {
    //     z.nodes[1].parent = this;
    // }
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
    // z.nodes[0].parent = this;
    z.nodes.push(y);
    // if (typeof y !== "number") {
    //     z.nodes[1].parent = this;
    // }
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
    // z.nodes[0].parent = this;
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
                if (typeof l === "number") {
                    l = torch.tensor(l);
                }
                if (typeof r === "number") {
                    r = torch.tensor(r);
                }
                return l.add(r);

                case SUB:
                var l = this.nodes[0].derive();
                var r = this.nodes[1].derive();
                // console.log('sub');
                // console.log(l);
                // console.log(r);
                if (typeof r === "number") {
                    r = torch.tensor(r);
                }
                if (typeof l === "number") {
                    l = torch.tensor(l);
                }
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
                return (l.mul(this.nodes[1])).add((r.mul(this.nodes[0])));

                case TRANSPOSE:
                var l = this.nodes[0].derive();
                return l;

                case POW:
                var l = this.nodes[0].derive();
                var r = this.nodes[1].derive();

                if (this.nodes[0].constant) {
                    if (this.nodes[1].rows == 0 && this.nodes[1].cols == 0) {
                        return r * Math.pow(this.nodes[0].mat, this.nodes[1].mat);
                    } else {
                        return this.nodes[0].pow(this.nodes[1]).mul(r);
                    }
                }
                break;

                case DIV:
                var l = this.nodes[0].derive();
                var r = this.nodes[1].derive();

                if (this.nodes[0].rows === 1 && this.nodes[0].cols > 1) {
                    // console.log("=========== in ============s");
                    var d_x = (this.nodes[1].mul(l).sub(this.nodes[0].mul(r))).div(this.nodes[1].pow(2));
                    var grad = v.mul(d_x);
                    // console.log(v.mul(d_x));
                    for (var i=0; i<this.nodes[0].cols; i++) {
                        var d_sum = this.nodes[0].mat[0][i];
                        // console.log(d_sum);
                        var d_x_i = this.nodes[0].mul(torch.const(d_sum)).div(this.nodes[1].pow(2)).minus();
                        // console.log(d_x_i.mat);
                        // console.log(v.mul(d_x_i));
                        let grad_ = v.mul(d_x_i);
                        for (var j=0; j<this.nodes[0].cols; j++) {
                            if (i !== j)
                                grad.mat[0][i] += grad_.mat[0][j];
                        }
                    }
                    // console.log(grad);
                    grad.isgrad = true;
                    return grad;
                }
                // console.log("div");
                // console.log(this.nodes[0]);
                // console.log(this.nodes[1]);
                // console.log(l);
                // console.log(r);
                // console.log('---');
                // console.log(this.nodes[1].mul(l));
                // console.log(this.nodes[0].mul(r));
                if (this.nodes[1].rows == 0 && this.nodes[1].cols == 0) {
                    return (l * this.nodes[1].mat - this.nodes[0].mat* r.mat) / Math.pow(this.nodes[1].mat, 2);
                } else {
                    return (this.nodes[1].mul(l).sub(this.nodes[0].mul(r))).div(this.nodes[1].pow(2));
                }
            
                case MINUS:
                var l = this.nodes[0].derive();
                return -1;

                case REDUCE_SUM:
                console.log("reduce");
                console.log(r);
                var r = this.nodes[0].derive();
                // var z = torch.tensor();
                // z.rows = r.rows;
                // z.cols = r.cols;
                // for (var i=0; i<r.rows; i++) {
                //     var row = [];
                //     for (var j=0; j<r.cols; j++) {
                //         row.push(r);
                //     }
                //     z.mat.push(row);
                // }
                return r;

                case MAX:
                var l = this.nodes[0];
                var r = this.nodes[1];
                // console.log(l);
                // console.log(r);
                return 1;
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

Tensor.prototype.findAllVaribles = function() {
    let vars = [];
    for (var i=0; i<this.nodes.length; i++) {
        if (this.nodes[i].constant === false && this.nodes[i].op === 0) {
            vars.push(this.nodes[i]);
        }
        let next = this.nodes[i].findAllVaribles();
        if (next.length > 0) {
            vars = next.concat(vars);
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

    static variable() {
        return new Tensor(arguments);
    }
    
    static function(head, name) {
        return new Function(head, name);
    }
}

class F {
    static sigmoid(x) {
        return torch.function(torch.const(1).div(torch.const(1).add(torch.const(Math.E).pow(torch.tensor(x).minus()))));
    }

    static relu(x) {
        let func  = torch.function(torch.tensor(x).max());
        func.name = "relu";
        return func;
    }

    static softmax(x) {
        let exp = torch.const(Math.E).pow(torch.tensor(x));
        let sum = exp.reduce_sum();
        let func = torch.function(exp.div(sum));
        func.name = "softmax";
        return func;
    }
}   

class Model {
    constructor(func) {
        this.func = func;
        this.parameters = [];
    }

    pred(x) {
        let result = this.func(x)
        this.parameters = result.parameters;
        return result.result;
    }
}

class Layer {
    constructor(func) {
        this.allocte = false;
        this.func = func;
        this.class = "layer";
        this.params_idx = [];
    }
}

class nn {
    static Linear(input_size, output_size) {
        return new Layer(function(x, parameters, idx) {
            if (this.allocte == false) {
                this.params_idx.push(parameters.length);
                parameters.push(torch.tensor(input_size, output_size));
                this.params_idx.push(parameters.length);
                parameters.push(torch.tensor(1, output_size));
                this.allocte = true;
            }

            let func = torch.function(torch.tensor(x).mul(parameters[this.params_idx[0]]).add(parameters[this.params_idx[1]]));
            func.name = "linear";
            return func;
        });
    }

    static ReLU() {
        return function(x, parameters, idx) {
            return F.relu(x);
        }
    }

    static Sigmoid() {
        return F.sigmoid;
    }

    static Softmax() {
        return F.softmax;
    }

    static MSELoss() {
        return function(y_, y) {
            let func = torch.function(torch.tensor(y_).sub(torch.const(y)));
            func.name = "MSELoss";
            return func;
        }
    }

    static Sequentail() {
        let args = arguments;
        let parameters = [];
        let func = function(x) {
            let last = x;
            for (let i=0; i<args.length; i++) {
                if ('class' in args[i]) {
                    last = args[i].func(last, parameters, i);
                } else {
                    last = args[i](last, parameters, i);
                }
            }
            return {"result":last, "parameters": parameters};
        }
        return new Model(func);
    }
}