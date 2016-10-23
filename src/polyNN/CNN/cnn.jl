# CNN

function get_index(p, u, R, r)
	p*u + R - r
end

@polly function cnn_forward(inp_F, W, out_F,nu,nv)
	nn,nk,np,nq = size(out_F)
	nc,nh,nw = size(inp_F)[2:end]
	nr,ns = size(W)[3:end]

	for n=1:nn, k=1:nk, p=1:np, q=1:nq, c=1:nc, r=1:nr, s=1:ns
		out_F[n,k,p,q] += W[k,c,r,s] * inp_F[n,c,get_index(p,nu,nr,r),get_index(q,nv,ns,s)]
	end

end

let
	nn = 50
	nk = 40
	nc = 75
	nr = 6
	ns = 6
	nw = 50
	nh = 50
	nu = 5
	nv = 5
	np = div((nh - nr),nu) + 1
	nq = div((nw - ns),nv) + 1
	out_F = zeros(Float32,nn,nk,np,nq)
	W = zeros(Float32,nk,nc,nr,ns)
	inp_F = zeros(Float32,nn,nc,nh,nw)
	err_in = zeros(Float32,nn,nc,nh,nw)
	err_out = zeros(Float32,nn,nk,np,nq)

#tic()
#cnn_forward(inp_F, W, out_F,nu,nv)
#run_time = toq()
#println("done (took $run_time) seconds")
	#init_array

	SUITE["cnn"] =@benchmarkable cnn_forward(out_F,W,inp_F) setup= (inp_F = copy($inp_F); W = copy($W); out_F = copy($out_F), nu, nv)
end