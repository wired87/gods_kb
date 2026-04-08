

from rest_framework.response import Response
from rest_framework.views import APIView

from ggoogle.bq.auth_handler import BQAuthHandler


bq_auth_handler = BQAuthHandler()







class QueryToSlGenesView(APIView):
    def post(self, request, *args, **kwargs):
        user_id = request.data.get("user_id")
        query = request.data.get("query")

        if not user_id or not query:
            return Response({"error": "Missing user_id or query"}, status=400)




        # Format the results
        results_list = [{"file": row.file, "url": row.url, "text": row.text, "distance": row.distance} for row in results]

        return Response({"results": results_list}, status=200)