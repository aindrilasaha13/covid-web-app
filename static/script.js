// api url
const api_url = [
	"http://127.0.0.1:5000/vaccination_item1",
	"http://127.0.0.1:5000/vaccination_item2", 
	"http://127.0.0.1:5000/vaccination_item3",
	"http://127.0.0.1:5000/vaccination_item4",
	"http://127.0.0.1:5000/vaccination_item5",
	"http://127.0.0.1:5000/vaccination_item6"];

// Defining async function
async function getapi(url) {
	
	var data=[];
	
	for(var i=0;i<6;i++)
	{
		response = await fetch(url[i]);
		data[i] = await response.json();
	}

	show(data);
}
// Calling that async function
getapi(api_url);

function show(data) {
	
	document.getElementById("text-1").innerHTML = data[0];
	document.getElementById("text-2").innerHTML = data[1];
	document.getElementById("text-3").innerHTML = data[2];
	document.getElementById("text-4").innerHTML = data[3];
	document.getElementById("text-5").innerHTML = data[4];
	document.getElementById("text-6").innerHTML = data[5];
}
