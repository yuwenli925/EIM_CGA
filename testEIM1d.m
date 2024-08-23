iter = 200;

k = 4; 
ntype = 'L2';
f = @(z, b) max(0, b(1).*z+b(2)).^k;
% f = @(z, b) (b(1).*z+b(2))>0;

% b = linspace(-2, 2, 500)'; 
% xset = linspace(0,1,1000)';

b = linspace(-2, 2, 1000)'; 
xset = linspace(0,1,1000)';

% b = linspace(-2, 2, 1000)'; 
% xset = linspace(0,1,10000)';

h = xset(2) - xset(1);
w = [-1, 1];  
[W, Bb] = meshgrid(w, b);  
muset = [W(:) Bb(:)]; 

musize = size(muset,1);
xsize = size(xset,1);
muid = zeros(iter,1);
xid = muid;
L = zeros(musize,1);
B = zeros(iter,iter);
Q = zeros(xsize,iter);

fx = zeros(xsize,musize);
for i = 1:musize
    fx(:,i) = f(xset, muset(i, :));
end

error = zeros(iter,1);
Lambda = error;
for n = 1:iter
    if n == 1
        if strcmp(ntype,'Linf')
            L = max(abs(fx));
        end
        if strcmp(ntype,'L2')
            L = h*sum(fx.*fx);
        end
    else
        r = fx - Q(:, 1:n-1)*( B(1:n-1, 1:n-1)\fx(xid(1:n-1),:) );
        if strcmp(ntype,'Linf')
            L = max(abs(r));
        end
        if strcmp(ntype,'L2')
            L = h*sum(r.*r);
        end
    end
    [~, muid(n)] = max(L);
    if n == 1
        r = fx(:,muid(n));
    else
        r = fx(:,muid(n)) - Q(:,1:n-1)*( B(1:n-1,1:n-1)\fx(xid(1:n-1),muid(n)) );
    end
    [error(n), xid(n)] = max(abs(r));
    
    if strcmp(ntype,'L2')
        error(n) = sqrt(h*sum(r.*r));
    end

    Q(:, n) = r/r(xid(n));
    B(n, 1:n) = Q(xid(n), 1:n);
    Lambda(n) = norm(Q,'inf');
    if mod(n,10) == 0
        fprintf('EIM at the %d-th step, error is %e\n',n,error(n));
    end
end

mu = muset(muid,:);
x = xset(xid,:);

N = 1:iter;
st = 50;
avgLambda = Lambda;
h = zeros(iter,1);
for i=1:iter
    avgLambda(i) = (prod(1+Lambda(1:i))).^(1/i);
    idx = (mu(1:i,1)==1);
    node = sort(mu(idx,2));
    if length(node)==1
        h(i)=1;
    else
        h(i) = min(node(2:end)-node(1:end-1));
    end
end

%error = error./(1+Lambda)./avgLambda;
temp = polyfit(log(N(st:end)),log(error(st:end)),1);
fprintf('The order of EIM error is %.3e \n', -temp(1));
loglog(N,error,'.',N,N.^(-k),'o');

