<script>let a=0,b=0,op='+',r=0;
function go(){r=op==='+'?a+b:op==='-'?a-b:op==='*'?a*b:b?a/b:0;}</script>
<main><input type=number bind:value={a}/><select bind:value={op}><option>+</option><option>-</option><option>*</option><option>/</option></select><input type=number bind:value={b}/><button on:click={go}>=</button><p>{r}</p><button on:click={()=>{a=0;b=0;r=0}}>clear</button></main>
