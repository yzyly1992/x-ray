import stripe
from django.core.mail import send_mail
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView
from django.views import View
from django.core.exceptions import ObjectDoesNotExist
from .models import Product

stripe.api_key = settings.STRIPE_SECRET_KEY

class SuccessView(TemplateView):
    template_name = "success.html"

class CancelView(TemplateView):
    template_name = "cancel.html"

class ProductLandingPageView(TemplateView):
    template_name= "landing.html"

    def get_context_data(self, **kwargs):
        context = super(ProductLandingPageView, self).get_context_data(**kwargs)
        
        try:
            product = Product.objects.get(name="Scans")
            context.update({
                "product": product,
                "STRIPE_PUBLIC_KEY": settings.STRIPE_PUBLIC_KEY,
                "product_available": True
            })
        except ObjectDoesNotExist:
            context.update({
                "product": None,
                "STRIPE_PUBLIC_KEY": settings.STRIPE_PUBLIC_KEY,
                "product_available": False,
                "error_message": "Products unavailable. Please set up your stripe product information in the database."
            })
        
        return context

# Create your views here.
class CreateCheckoutSessionView(View):
    def post (self, request, *args, **kwargs):
        product_id = self.kwargs["pk"]
        try:
            product = Product.objects.get(id=product_id)
        except ObjectDoesNotExist:
            return JsonResponse({
                'error': 'Product not found'
            }, status=404)
            
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[
                {
                    'price_data': {
                        'currency': 'usd',
                        'unit_amount': product.price,
                        'product_data': {
                            'name': product.name
                        },
                    },
                    'quantity': 1,
                },
            ],
            metadata = {
                "product_id": product_id
            },
            mode='payment',
            success_url=settings.DOMAIN + 'success/',
            cancel_url=settings.DOMAIN + 'cancel/'
        )
        return JsonResponse({
            'id': checkout_session.stripe_id
        })

# Set your secret key. Remember to switch to your live secret key in production.
# See your keys here: https://dashboard.stripe.com/apikeys

@csrf_exempt
def stripe_webhook(request):
    payload = request.body  
    sig_header = request.META['HTTP_STRIPE_SIGNATURE']
    event = None
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        # Invalid payload
        return HttpResponse(status=400)
    except stripe.error.SignatureVerificationError as e:
      # Invalid signature
        return HttpResponse(status=400)

    if event['type'] == 'checkout.session.completed':
        session = stripe.checkout.Session.retrieve(
            event['data']['object']['id'],
            expand=['line_items'],
        )
        print(session['customer_details']['email'])
        
        customer_email = session['customer_details']['email']
        product_id = session['metadata']['product_id']

        try:
            product = Product.objects.get(id=product_id)
            send_mail(
                subject="Here is your product",
                message=f"Thanks for your purchase. The URL is {product.url}",
                recipient_list=[customer_email],
                from_email = "NCU@test.com"
            )
        except ObjectDoesNotExist:
            print(f"Product with id {product_id} not found")
            # You might want to log this error or handle it differently


    # For now, you only need to print out the webhook payload so you can see
    # the structure.
    return HttpResponse(status=200)



