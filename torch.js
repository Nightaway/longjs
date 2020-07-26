"use strict";

const ADD = 'add';
const MUL = 'mul';
const MATMUL = 'matmul';
const TRANSPOSE = 'transpose';
const POW = 'pow';
const LOG = 'log';
const SUB = 'sub';
const DIV = 'div';
const MINUS = 'minus';
const ReLU = 'relu';
const REDUCE_SUM = 'reduce_sum';

const OP = 'op';
const TENSOR = 'tensor';

function Function(head, name) {
    this.head = head;
    this.vars = head.findAllVaribles();
    this.grad = null;
    this.class = 'function';
    this.name = name;
}

Function.prototype.backward = function () {
    for (var i=0; i<this.vars.length; i++) {
        this.setvar(this.vars[i]);
        var d = this.head.derive(arguments[0]);
        this.unsetvar(this.vars[i]);

        console.log(d);

        var grad = this.head.matmul(d);

        console.log(grad);
        this.vars[i].grad = grad;
        console.log(this.vars[i]);
        if (this.vars[i].func !== null) {
            this.vars[i].func.backward(this.vars[i].grad);
        }
    }
}

Function.prototype.setvar = function(v) {
    for (var i=0; i<this.vars.length; i++) {
        if (this.vars[i] !== v) {
            this.vars[i].constant = true;
        }
    }
}

Function.prototype.unsetvar = function () {
    for (var i=0; i<this.vars.length; i++) {
        this.vars[i].constant = false;
    }
}

Function.prototype.findAllVaribles = function () {
    return this.head.findAllVaribles();
}

function Tensor() {
    var args = arguments[0];
    this.grad = null;
    this.nodes = [];
    this.type = TENSOR;
    this.op = 0;
    this.isgrad = false;
    this.constant = false;
    this.class = 'tensor';
    this.func = null;
    this.name = "";
    this.shape = [];
    this.darray = [];
    
   if (args.length === 1) {
        if (Array.isArray(args[0])) {
            var dim = [];
            var arg = args[0][0];
            while (Array.isArray(arg)) {
                dim.push(arg.length);
                arg = arg[0];
            }
            this.shape = [args[0].length];
            for (var i=0; i<dim.length; i++) {
                this.shape.push(dim[i]);
            }
            this.darray = args[0];
        } else if (typeof args[0] == "object") {
            if (args[0].class == "tensor") {
                this.grad = args[0].grad;
                this.nodes = args[0].nodes;
                this.type = args[0].type;
                this.op = args[0].op;
                this.constant = args[0].constant;
                this.name = args[0].name;
                this.shape = args[0].shape;
                this.darray = args[0].darray;
            } else if (args[0].class == "function") {
                this.shape = args[0].head.shape;
                this.darray = args[0].head.darray;
                this.func = args[0];
                // this.op = args[0].head.op;
                // this.type = args[0].head.type;
                // this.name = args[0].head.name;
            }
        } else {
            throw new Error('Tensor Init Error');
        }
    } else if (args.length === 2) {
        this.shape = args[0];
        var prev = null;
        var len = args[0].length;
        for (var j=args[0][len-1]; j>0; j--) {
            var row  = [];
            if (prev == null) {
                for (var k=0; k<j; k++) {
                    row.push(args[1]);
                }
                prev = row;
            } else {
                for (var k=0; k<j; k++) {
                    row.push(prev);
                }
                prev = row;
            }
        }
        this.darray = prev;
    } else if (args.length === 0) {
        // console.log('Arg len 0');
    } else {
        throw new Error('Tensor Init Error');
    }
}

function iter(darray, func, post) {
    if (darray.length > 0 && Array.isArray(darray[0])) {
        var arr = [];
        for (var i=0; i<darray.length; i++) {
            arr.push(iter(darray[i], func, post));
        }
        return arr;
        
    } else {
        var arr = [];
        for (var i=0; i<darray.length; i++) {
            arr.push(func(darray[i]));
        }

        if (post != null && post != undefined) {
            arr = post(arr);
        }
        return arr;
    }
}

Tensor.prototype.ReLU = function() {
    let z = torch.const();

    function _relu(e) {
        if (e <= 0) return 0;
        return e;
    }

    var darray = iter(this.darray, _relu);
    z.darray = darray;
    z.shape = this.shape;
    z.nodes.push(this);
    z.type = OP;
    z.op = ReLU;
    return z;
}

function ident(e) {
    return e;
}

Tensor.prototype.reduce_sum = function () {
    function _reduce_sum(a) {
        var sum = 0;
        for (var i=0; i<a.length; i++) {
            sum += a[i];
        }
        return sum;
    }

    var darray = iter(this.darray, ident, _reduce_sum);
    var z = torch.tensor(darray);
    z.darray = darray;
    z.shape = z.shape;
    z.nodes.push(this);
    z.type = OP;
    z.op = REDUCE_SUM;
    return z;
}

Tensor.prototype.transpose = function () {
    let z = torch.const();
    var darray = [];
    for (var i=0; i<this.shape[1]; i++) {
        var rows = [];
        for (var j=0; j<this.shape[0]; j++) {
            rows.push(this.darray[j][i]);
        }
        darray.push(rows);
    }
    z.darray = darray;
    z.shape[0] = this.shape[1];
    z.shape[1] = this.shape[0];
    z.nodes.push(this);
    z.type = 1;
    z.op = TRANSPOSE;
    return z;
}

Tensor.prototype.mul = function (y) {
    if (typeof y !== "number") {
        throw new Error('Mul oprand2 must is Number');
    }

    let z = torch.const();
    function _mul(e) {
        e = e * y;
        return e;
    }

    var darray = iter(this.darray, _mul);
    z.darray = darray;
    z.shape = this.shape;
    z.nodes.push(this);
    z.type = OP;
    z.op = MUL;
    return z;
}

Tensor.prototype.matmul = function (y) {
    if (typeof y !== "object") {
        throw new Error('MatMul oprand2 must is Object');
    }

    if (y.class != 'tensor') {
        throw new Error('MatMul oprand2 must is Tensor');
    }

    if (this.shape[1] != y.shape[0]) {
        throw new Error('MatMul oprand1 and oprand2 rank is Mismatch');
    }

    var z = torch.tensor();
    var darray = [];
    for (var i=0; i<this.shape[0]; i++) {
        var cols = [];
        for (var j=0; j<y.shape[1]; j++) {
            var col = [];
            for (var k=0; k<y.shape[0]; k++) {
                col.push(y.darray[k][j]);
            }
            cols.push(col);
        }
        var row = [];
        for (var j=0; j<cols.length; j++) {
            var sum = 0;
            for (var k=0; k<cols[j].length; k++) {
                sum += this.darray[i][k] * cols[j][k];
            }
            row.push(sum);
        }
        darray.push(row);
    }

    z.darray = darray;
    z.shape[0] = this.shape[0];
    z.shape[1] = y.shape[1];
    z.nodes.push(this);
    z.nodes.push(y);
    z.type = OP;
    z.op = MATMUL;
    return z;
}

Tensor.prototype.sub = function (y) {
    if (typeof y !== "object" && y.class != "tensor") {
        throw new Error('Sub oprand2 must is tensor');
    }

    var z = torch.tensor();
    function _sub(e) {
        e = e - y.darray[0][0];
        return e;
    }

    var darray = iter(this.darray, _sub);
    z.darray = darray;
    z.shape = this.shape;
    z.nodes.push(this);
    z.nodes.push(y);
    z.type = OP;
    z.op = SUB;
    return z;
}

Tensor.prototype.div = function (y) {
    if (typeof y !== "number") {
        throw new Error('Div oprand2 must is Number');
    }

    var z = torch.tensor();
    function _div(e) {
        e = e / y;
        return e;
    }

    var darray = iter(this.darray, _div);
    z.darray = darray;
    z.shape = this.shape;
    z.nodes.push(this);
    z.nodes.push(y);
    z.type = OP;
    z.op = DIV;
    return z;
}

Tensor.prototype.pow = function (y) {
    if (typeof y !== "number") {
        throw new Error('Pow oprand2 must is Number');
    }

    var z = torch.tensor();
    function _pow(e) {
        e = math.pow(e, y);
        return e;
    }

    var darray = iter(this.darray, _pow);
    z.darray = darray;
    z.shape = this.shape;
    z.nodes.push(this);
    z.nodes.push(y);
    z.type = OP;
    z.op = POW;
    return z;
}

Tensor.prototype.log = function () {
    if (typeof y !== "number") {
        throw new Error('Log oprand2 must is Number');
    }

    var z = torch.tensor();
    function _log(e) {
        e = math.log(e);
        return e;
    }

    var darray = iter(this.darray, _log);
    z.darray = darray;
    z.shape = this.shape;
    z.nodes.push(this);
    z.type = OP;
    z.op = POW;
    return z;
}

Tensor.prototype.add = function (y) {
    if (typeof y !== "object" && y.class != "tensor") {
        throw new Error('Add oprand2 must is tensor');
    }
    var z = torch.tensor();
    function _add(e) {
        e = e + y.darray[0][0];
        return e;
    }

    var darray = iter(this.darray, _add);
    z.darray = darray;
    z.shape = this.shape;
    z.nodes.push(this);
    z.nodes.push(y);
    z.type = OP;
    z.op = ADD;
    return z;
}

Tensor.prototype.sub_ = function (y) {
    if (typeof y !== "object" && y.class != "tensor") {
        throw new Error('Sub oprand2 must is tensor');
    }

    function _sub(e) {
        e = e - y.darray[0][0];
        return e;
    }

    var darray = iter(this.darray, _sub);
    this.darray = darray;
}

Tensor.prototype.minus = function () {
    var z = torch.tensor();
    function _minus(e) {
        e = -e;
        return e;
    }

    var darray = iter(this.darray, _minus);
    z.darray = darray;
    z.shape = this.shape;
    z.nodes.push(this);
    z.type = OP;
    z.op = MINUS;
    return z;
}

Tensor.prototype.derive = function(v) {
    if (this.constant) {
        return torch.tensor([[0]]);
    }
    if (this.func !== null) {
        return torch.tensor([[1]]);
    }

    switch (this.type) {
        case OP:
            switch(this.op) {
                case ADD:
                var l = this.nodes[0].derive();
                var r = this.nodes[1].derive();
                return l.add(r);

                case SUB:
                var l = this.nodes[0].derive();
                var r = this.nodes[1].derive();
                return l.sub(r);

                case MATMUL:
                var l = this.nodes[0].derive();
                var r = this.nodes[1].derive();
                return this.nodes[1].matmul(l).add(r.matmul(this.nodes[0]));

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
                if (typeof r === "number") {
                    r = torch.tensor(r);
                }
                if (typeof l === "number") {
                    l = torch.tensor(l);
                }
                if (this.nodes[0].constant) {
                    if (this.nodes[1].rows == 0 && this.nodes[1].cols == 0) {
                        return r.mat * Math.pow(this.nodes[0].mat, this.nodes[1].mat);
                    } else {
                        return this.nodes[0].pow(this.nodes[1]).mul(r);
                    }
                } else {
                    // return this.nodes[0].pow(this.nodes[1]).mul(r);
                    return this.mul(r.sub(torch.const(1)));
                }
                break;

                case DIV:
                var l = this.nodes[0].derive();
                var r = this.nodes[1].derive();
                if (typeof r === "number") {
                    r = torch.const(r);
                }
                if (typeof l === "number") {
                    l = torch.const(l);
                }

                if (this.nodes[0].rows === 1 && this.nodes[0].cols > 1) {
                    var d_x = (this.nodes[1].mul(l).sub(this.nodes[0].mul(r))).div(this.nodes[1].pow(2));
                    var grad = v.mul(d_x);
                    for (var i=0; i<this.nodes[0].cols; i++) {
                        var d_sum = this.nodes[0].mat[0][i];
                        var d_x_i = this.nodes[0].mul(torch.const(d_sum)).div(this.nodes[1].pow(2)).minus();
                        let grad_ = v.mul(d_x_i);
                        for (var j=0; j<this.nodes[0].cols; j++) {
                            if (i !== j)
                                grad.mat[0][i] += grad_.mat[0][j];
                        }
                    }
                    grad.isgrad = true;
                    return grad;
                }
                
                if (this.nodes[1].rows == 0 && this.nodes[1].cols == 0) {
                    if (this.nodes[0].rows == 0 && this.nodes[0].cols == 0) {
                        return (l.mat * this.nodes[1].mat - this.nodes[0].mat * r.mat) / Math.pow(this.nodes[1].mat, 2);
                    } else {
                        return (l * this.nodes[1].mat - this.nodes[0].mat* r.mat) / Math.pow(this.nodes[1].mat, 2);
                    }
                } else {
                    return (this.nodes[1].mul(l).sub(this.nodes[0].mul(r))).div(this.nodes[1].pow(2));
                }
            
                case MINUS:
                var l = this.nodes[0].derive();
                return -1;

                case REDUCE_SUM:
                var r = this.nodes[0].derive();
                return r;

                // case ReLU:
                // var l = this.nodes[0];
                // return 1;
            }
        break;

        case TENSOR:
            if (this.constant) {
                return torch.tensor([[0]]);
            } else if (this.function !== null) {
                return torch.tensor([[1]]);
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
        let func = torch.function(torch.const(1).div(torch.const(1).add(torch.const(Math.E).pow(torch.tensor(x).minus()))));
        func.name = "Sigmoid";
        return func;
    }

    static relu(x) {
        let func  = torch.function(torch.tensor(x).ReLU());
        func.name = "ReLU";
        return func;
    }

    static softmax(x) {
        let exp = torch.const(Math.E).pow(torch.tensor(x));
        let sum = exp.reduce_sum();
        let func = torch.function(exp.div(sum));
        func.name = "Softmax";
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

    static MSELoss1() {
        return function(y_, y) {
            let func = torch.function(torch.tensor(y_).sub(torch.const(y)));
            func.name = "MSELoss1";
            return func;
        }
    }

    static MSELoss() {
        return function(y_, y) {
            let func = torch.function(torch.tensor(y_).sub(torch.const(y)).pow(torch.const(2)));
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