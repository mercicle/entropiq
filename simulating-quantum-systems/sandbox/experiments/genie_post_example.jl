using Genie, Genie.Router, Genie.Renderer.Json, Genie.Requests
using HTTP

route("/echo", method = POST) do
  message = jsonpayload()
  (:echo => (message["message"] * " ") ^ message["repeat"]) |> json
end

route("/send") do
  response = HTTP.request("POST", "http://localhost:8000/echo", [("Content-Type", "application/json")], """{"message":"hello", "repeat":3}""")
  response.body |> String |> json
end

route("/customers/:customer_id/orders/:order_id") do
  "You asked for the order $(payload(:order_id)) for customer $(payload(:customer_id))"
end

Genie.startup(async = false)
