const form=document.getElementById("risk-form");
const submitBtn=document.getElementById("submit-btn");
const idle=document.getElementById("idle");
const loading=document.getElementById("loading");
const output=document.getElementById("output");
const error=document.getElementById("error");

const examples=[
{person_age:30,person_income:60000,person_home_ownership:"OWN",person_emp_length:8,loan_intent:"EDUCATION",loan_grade:"A",loan_amnt:5000,loan_int_rate:7.5,loan_percent_income:.08,cb_person_default_on_file:"N",cb_person_cred_hist_length:10},
{person_age:24,person_income:28000,person_home_ownership:"RENT",person_emp_length:1,loan_intent:"PERSONAL",loan_grade:"D",loan_amnt:10000,loan_int_rate:17.5,loan_percent_income:.36,cb_person_default_on_file:"Y",cb_person_cred_hist_length:3},
{person_age:35,person_income:55000,person_home_ownership:"MORTGAGE",person_emp_length:6,loan_intent:"HOMEIMPROVEMENT",loan_grade:"B",loan_amnt:12000,loan_int_rate:10.5,loan_percent_income:.22,cb_person_default_on_file:"N",cb_person_cred_hist_length:9}
];

function setState(s){[idle,loading,output,error].forEach(x=>x.classList.add("hidden"));s.classList.remove("hidden")}
function getData(){
 const d=new FormData(form);
 return {
  person_age:+d.get("person_age"),person_income:+d.get("person_income"),
  person_home_ownership:d.get("person_home_ownership"),person_emp_length:+d.get("person_emp_length"),
  loan_intent:d.get("loan_intent"),loan_grade:d.get("loan_grade"),loan_amnt:+d.get("loan_amnt"),
  loan_int_rate:+d.get("loan_int_rate"),loan_percent_income:+d.get("loan_percent_income"),
  cb_person_default_on_file:d.get("cb_person_default_on_file"),cb_person_cred_hist_length:+d.get("cb_person_cred_hist_length")
 };
}
function fillForm(x){Object.entries(x).forEach(([k,v])=>{if(form.elements[k])form.elements[k].value=v})}
function animateNumber(el,target){
 const start=performance.now();
 function frame(now){
  const p=Math.min((now-start)/900,1),e=1-Math.pow(1-p,3);
  el.textContent=(target*e*100).toFixed(1)+"%";
  if(p<1)requestAnimationFrame(frame);
 }
 requestAnimationFrame(frame);
}
function showResult(d){
 const p=+d.default_probability,t=+d.threshold,high=+d.default_prediction===1;
 setState(output);
 document.getElementById("result-time").textContent=new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"});
 document.getElementById("risk-label").textContent=d.Result;
 const badge=document.getElementById("risk-badge"),fill=document.getElementById("meter-fill"),icon=document.getElementById("decision-icon");
 badge.textContent=high?"HIGH":"LOW";badge.classList.toggle("high",high);
 fill.classList.toggle("high",high);icon.classList.toggle("high",high);icon.textContent=high?"!":"✓";
 document.getElementById("threshold-marker").style.left=`${t*100}%`;
 document.getElementById("threshold-text").textContent=`Threshold ${(t*100).toFixed(1)}%`;
 document.getElementById("decision-text").textContent=high?"The predicted probability is above the configured decision threshold.":"The predicted probability is below the configured decision threshold.";
 requestAnimationFrame(()=>{fill.style.width=`${Math.min(100,p*100)}%`;animateNumber(document.getElementById("probability"),p)});
}
form.addEventListener("submit",async e=>{
 e.preventDefault();
 if(!form.reportValidity())return;
 submitBtn.disabled=true;submitBtn.classList.add("loading");setState(loading);
 document.getElementById("result-time").textContent="Processing";
 try{
  const r=await fetch("/predict/",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(getData())});
  if(!r.ok){let msg=`API returned HTTP ${r.status}.`;try{const d=await r.json();if(d.detail)msg=typeof d.detail==="string"?d.detail:"Please check the submitted values."}catch(_){}throw new Error(msg)}
  showResult(await r.json());
 }catch(err){
  setState(error);
  document.getElementById("result-time").textContent="Request failed";
  document.getElementById("error-text").textContent=err.message||"Could not reach the prediction API.";
 }finally{submitBtn.disabled=false;submitBtn.classList.remove("loading")}
});
document.getElementById("example-btn").onclick=()=>{
 fillForm(examples[Math.floor(Math.random()*examples.length)]);
 const b=document.getElementById("example-btn");b.textContent="Example loaded";setTimeout(()=>b.textContent="Load example",1100);
};
document.getElementById("reset-btn").onclick=()=>{
 setState(idle);document.getElementById("result-time").textContent="Awaiting input";document.getElementById("meter-fill").style.width="0%";
};
document.getElementById("retry-btn").onclick=()=>form.requestSubmit();
