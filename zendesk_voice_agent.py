import sounddevice as sd
import numpy as np
import websocket
import threading
import json
import urllib.parse
import time
import google.generativeai as genai
import queue
import requests
import io
import wave
import aiohttp
import asyncio
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Your API keys and URLs
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

# Zendesk Configuration
ZENDESK_SUBDOMAIN = os.environ.get("ZENDESK_SUBDOMAIN")
ZENDESK_API_TOKEN = os.environ.get("ZENDESK_API_TOKEN")
ZENDESK_ADMIN_EMAIL = os.environ.get("ZENDESK_ADMIN_EMAIL")

DEEPGRAM_STT_URL = (
    "wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1"
    "&punctuate=true&interim_results=true&smart_format=true&model=nova-3&utterance_end_ms=1000"
    f"&keyterm={urllib.parse.quote('Jeff:10 Gmail:10 hewhocodes:10 Peter:10 peter:10 Pee-ter:10')}"
)

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Global variables
conversation_history = []
ws_stt = None
stt_lock = threading.Lock()
audio_queue = queue.Queue()
is_speaking = threading.Event()
transcript_buffer = []
conversation_active = False
conversation_topic = ""
customer_email = None
customer_data = None
ticket_context = None
current_ticket_id = None
# New variables for email confirmation flow
pending_ticket_request = None
email_confirmation_pending = False
# Track last user intent for context
last_intent = None
last_intent_details = None
# Track last referenced ticket/order/appointment for context
last_ticket = None
last_order = None
last_appointment = None

# Zendesk API class
class ZendeskAPI:
    def __init__(self, subdomain: str, api_token: str, admin_email: str):
        self.base_url = f"https://{subdomain}.zendesk.com/api/v2"
        self.auth = aiohttp.BasicAuth(login=f"{admin_email}/token", password=api_token)
        self.headers = {"Content-Type": "application/json"}
    
    async def get_customer_by_email(self, email: str) -> Optional[Dict]:
        try:
            url = f"{self.base_url}/users/search?query=email:{email}"
            print(f"🔍 Zendesk API call: {url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, auth=self.auth, headers=self.headers) as resp:
                    print(f"🔍 Zendesk response status: {resp.status}")
                    if resp.status != 200:
                        error_text = await resp.text()
                        print(f"🔍 Zendesk error response: {error_text}")
                        return None
                    data = await resp.json()
                    print(f"🔍 Zendesk response data: {data}")
                    users = data.get("users", [])
                    print(f"🔍 Found {len(users)} users")
                    if users:
                        return users[0]
            return None
        except Exception as e:
            print(f"Zendesk get_customer_by_email error: {e}")
            return None
    
    async def get_open_tickets_by_email(self, email: str) -> List[Dict]:
        try:
            url = f"{self.base_url}/search?query=type:ticket requester:{email} status:open"
            print(f"🎫 Zendesk ticket search: {url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, auth=self.auth, headers=self.headers) as resp:
                    print(f"🎫 Zendesk ticket response status: {resp.status}")
                    if resp.status != 200:
                        error_text = await resp.text()
                        print(f"🎫 Zendesk ticket error: {error_text}")
                        return []
                    data = await resp.json()
                    print(f"🎫 Zendesk ticket response: {data}")
                    tickets = data.get("results", [])
                    print(f"🎫 Found {len(tickets)} tickets")
                    return tickets
        except Exception as e:
            print(f"Zendesk get_open_tickets_by_email error: {e}")
            return []
    
    async def create_ticket(self, subject: str, comment: str, requester_email: str, requester_name: str = None, tags: List[str] = None, priority: str = "normal") -> Optional[str]:
        try:
            url = f"{self.base_url}/tickets"
            payload = {
                "ticket": {
                    "subject": subject,
                    "comment": {
                        "body": comment,
                        "public": True
                    },
                    "requester": {
                        "email": requester_email,
                        "name": requester_name or "Customer"
                    },
                    "tags": tags or [],
                    "priority": priority
                }
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, auth=self.auth, headers=self.headers, json=payload) as resp:
                    if resp.status not in [200, 201]:
                        print(f"Zendesk create_ticket error: {resp.status} - {await resp.text()}")
                        return None
                    result = await resp.json()
                    return str(result.get("ticket", {}).get("id"))
        except Exception as e:
            print(f"Zendesk create_ticket error: {e}")
            return None

    async def add_comment_to_ticket(self, ticket_id: str, comment: str, public: bool = True) -> bool:
        try:
            url = f"{self.base_url}/tickets/{ticket_id}"
            payload = {
                "ticket": {
                    "comment": {
                        "body": comment,
                        "public": public
                    }
                }
            }
            async with aiohttp.ClientSession() as session:
                async with session.put(url, auth=self.auth, headers=self.headers, json=payload) as resp:
                    return resp.status in [200, 201]
        except Exception as e:
            print(f"Zendesk add_comment_to_ticket error: {e}")
            return False

    async def merge_tickets(self, main_ticket_id: str, ticket_ids: List[str]) -> bool:
        try:
            url = f"{self.base_url}/tickets/{main_ticket_id}/merge"
            payload = {"ids": ticket_ids}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, auth=self.auth, headers=self.headers, json=payload) as resp:
                    return resp.status in [200, 201]
        except Exception as e:
            print(f"Zendesk merge_tickets error: {e}")
            return False

    async def create_customer(self, email: str, name: str = None, phone: str = None) -> Optional[Dict]:
        """Create a new customer in Zendesk"""
        try:
            url = f"{self.base_url}/users"
            payload = {
                "user": {
                    "email": email,
                    "name": name or "Customer",
                    "phone": phone,
                    "role": "end-user",
                    "verified": True
                }
            }
            print(f"🔍 Creating customer in Zendesk: {email}")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, auth=self.auth, headers=self.headers, json=payload) as resp:
                    print(f"🔍 Create customer response status: {resp.status}")
                    if resp.status not in [200, 201]:
                        error_text = await resp.text()
                        print(f"🔍 Create customer error: {error_text}")
                        return None
                    result = await resp.json()
                    print(f"🔍 Customer created successfully: {result}")
                    return result.get("user")
        except Exception as e:
            print(f"Zendesk create_customer error: {e}")
            return None

# Initialize Zendesk API
zendesk = ZendeskAPI(ZENDESK_SUBDOMAIN, ZENDESK_API_TOKEN, ZENDESK_ADMIN_EMAIL)

# Function definitions for the LLM
ZENDESK_FUNCTIONS = [
    {
        "name": "get_customer_info",
        "description": "Get customer information from Zendesk by email address",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "Customer's email address"
                }
            },
            "required": ["email"]
        }
    },
    {
        "name": "search_customer_tickets",
        "description": "Search for customer's tickets, appointments, or orders in Zendesk",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "Customer's email address"
                },
                "search_type": {
                    "type": "string",
                    "enum": ["tickets", "appointments", "orders", "all"],
                    "description": "Type of items to search for"
                }
            },
            "required": ["email", "search_type"]
        }
    },
    {
        "name": "create_support_ticket",
        "description": "Create a new support ticket in Zendesk for customer issues",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "Customer's email address"
                },
                "subject": {
                    "type": "string",
                    "description": "Ticket subject"
                },
                "description": {
                    "type": "string",
                    "description": "Ticket description"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"],
                    "description": "Ticket priority"
                }
            },
            "required": ["email", "subject", "description"]
        }
    },
    {
        "name": "confirm_email_and_create_ticket",
        "description": "Confirm email address and create support ticket for new customers",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "Customer's email address to confirm"
                },
                "subject": {
                    "type": "string",
                    "description": "Ticket subject"
                },
                "description": {
                    "type": "string",
                    "description": "Ticket description"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"],
                    "description": "Ticket priority"
                }
            },
            "required": ["email", "subject", "description"]
        }
    },
    {
        "name": "add_comment_to_ticket",
        "description": "Add a comment to an existing ticket",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "The ticket ID to add comment to"
                },
                "comment": {
                    "type": "string",
                    "description": "The comment to add"
                },
                "public": {
                    "type": "boolean",
                    "description": "Whether the comment should be public or internal"
                }
            },
            "required": ["ticket_id", "comment"]
        }
    },
    {
        "name": "escalate_to_billing",
        "description": "Escalate a customer request to the billing department",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "Customer's email address"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for escalation"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"],
                    "description": "Priority level for the escalation"
                }
            },
            "required": ["email", "reason"]
        }
    }
]

# Function implementations
async def execute_zendesk_function(function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Zendesk functions based on LLM function calls"""
    try:
        if function_name == "get_customer_info":
            email = arguments.get("email")
            if not email:
                return {"error": "Email is required"}
            
            customer = await zendesk.get_customer_by_email(email)
            if customer:
                return {
                    "success": True,
                    "customer": {
                        "id": customer.get("id"),
                        "name": customer.get("name"),
                        "email": customer.get("email"),
                        "phone": customer.get("phone"),
                        "created_at": customer.get("created_at")
                    }
                }
            else:
                return {"success": False, "error": "Customer not found"}
        
        elif function_name == "search_customer_tickets":
            email = arguments.get("email")
            search_type = arguments.get("search_type", "all")
            
            if not email:
                return {"error": "Email is required"}
            
            # Get all tickets for the customer
            tickets = await zendesk.get_open_tickets_by_email(email)
            
            if search_type == "appointments":
                # Filter for appointment-related tickets (check subject AND description)
                appointments = []
                for ticket in tickets:
                    subject = ticket.get("subject", "").lower()
                    description = ticket.get("description", "").lower()
                    
                    # Check if it's appointment-related
                    appointment_keywords = ["appointment", "meeting", "schedule", "booking", "call", "evisa", "visa"]
                    if any(keyword in subject for keyword in appointment_keywords) or any(keyword in description for keyword in appointment_keywords):
                        appointments.append(ticket)
                
                return {"success": True, "appointments": appointments}
            
            elif search_type == "orders":
                # Filter for order-related tickets
                orders = [t for t in tickets if any(keyword in t.get("subject", "").lower() 
                                                   for keyword in ["order", "purchase", "payment", "invoice"])]
                return {"success": True, "orders": orders}
            
            else:
                return {"success": True, "tickets": tickets}
        
        elif function_name == "create_support_ticket":
            email = arguments.get("email")
            subject = arguments.get("subject")
            description = arguments.get("description")
            priority = arguments.get("priority", "normal")
            
            if not all([email, subject, description]):
                return {"error": "Email, subject, and description are required"}
            
            # Get customer info first
            customer = await zendesk.get_customer_by_email(email)
            if not customer:
                # Customer doesn't exist - return error to trigger email confirmation
                return {"error": "Customer not found", "needs_confirmation": True, "email": email}
            
            # Check for existing open tickets
            open_tickets = await zendesk.get_open_tickets_by_email(email)
            if open_tickets:
                # Add internal note to existing ticket
                ticket_id = str(open_tickets[0]["id"])
                success = await zendesk.add_comment_to_ticket(ticket_id, description, public=False)
                if success:
                    return {
                        "success": True,
                        "action": "note_added",
                        "ticket_id": ticket_id,
                        "message": f"Your request has been added to your existing ticket."
                    }
                else:
                    return {"error": "Failed to add note to existing ticket"}
            else:
                # Create new ticket
                ticket_id = await zendesk.create_ticket(
                    subject=subject,
                    comment=description,
                    requester_email=email,
                    requester_name=customer.get("name"),
                    tags=["voice_chat", "support"],
                    priority=priority
                )
                if ticket_id:
                    return {
                        "success": True,
                        "action": "ticket_created",
                        "ticket_id": ticket_id,
                        "message": f"Your request has been submitted successfully."
                    }
                else:
                    return {"error": "Failed to create ticket"}
        
        elif function_name == "confirm_email_and_create_ticket":
            email = arguments.get("email")
            subject = arguments.get("subject")
            description = arguments.get("description")
            priority = arguments.get("priority", "normal")
            
            if not all([email, subject, description]):
                return {"error": "Email, subject, and description are required"}
            
            # Create the customer first
            customer = await zendesk.create_customer(email)
            if not customer:
                return {"error": "Failed to create customer account"}
            
            # Now create the ticket
            ticket_id = await zendesk.create_ticket(
                subject=subject,
                comment=description,
                requester_email=email,
                requester_name=customer.get("name"),
                tags=["voice_chat", "support", "new_customer"],
                priority=priority
            )
            if ticket_id:
                return {
                    "success": True,
                    "action": "customer_and_ticket_created",
                    "ticket_id": ticket_id,
                    "customer_id": customer.get("id"),
                    "message": f"Great! I've created your account and submitted your request. Our team will review it and get back to you soon."
                }
            else:
                return {"error": "Failed to create ticket"}
        
        elif function_name == "add_comment_to_ticket":
            ticket_id = arguments.get("ticket_id")
            comment = arguments.get("comment")
            public = arguments.get("public", True)
            
            if not all([ticket_id, comment]):
                return {"error": "Ticket ID and comment are required"}
            
            success = await zendesk.add_comment_to_ticket(ticket_id, comment, public)
            if success:
                return {
                    "success": True,
                    "message": f"Comment added to ticket #{ticket_id}"
                }
            else:
                return {"error": "Failed to add comment"}
        
        elif function_name == "escalate_to_billing":
            email = arguments.get("email")
            reason = arguments.get("reason")
            priority = arguments.get("priority", "normal")
            
            if not all([email, reason]):
                return {"error": "Email and reason are required"}
            
            # Find or create a ticket for the customer
            customer = await zendesk.get_customer_by_email(email)
            if not customer:
                return {"error": "Customer not found for escalation"}
            
            open_tickets = await zendesk.get_open_tickets_by_email(email)
            if open_tickets:
                ticket_id = str(open_tickets[0]["id"])
                # Add internal note
                note_success = await zendesk.add_comment_to_ticket(ticket_id, f"Escalation to billing: {reason}", public=False)
                if not note_success:
                    return {"error": "Failed to add escalation note to ticket"}
            else:
                # No open ticket, create one
                ticket_id = await zendesk.create_ticket(
                    subject="Billing Escalation Request",
                    comment=f"Escalation to billing: {reason}",
                    requester_email=email,
                    requester_name=customer.get("name"),
                    tags=["voice_chat", "support", "escalation"],
                    priority=priority
                )
                if not ticket_id:
                    return {"error": "Failed to create ticket for escalation"}
            
            return {"success": True, "message": "Your request has been escalated to our billing department. They will review your case and contact you within 24 hours."}
        
        else:
            return {"error": f"Unknown function: {function_name}"}
    
    except Exception as e:
        print(f"Function execution error: {e}")
        return {"error": f"Function execution failed: {str(e)}"}

def build_function_calling_prompt(transcript: str, conversation_history: List[Dict], customer_data: Optional[Dict] = None) -> str:
    """Build a prompt that includes function calling instructions with smart conversation flow"""
    
    # Determine conversation phase
    has_email = customer_data is not None
    recent_messages = conversation_history[-4:] if conversation_history else []
    
    prompt = """You are a helpful customer service assistant. Follow this conversation flow:

CONVERSATION PHASES:
1. GENERAL CONVERSATION: For greetings, general questions, basic help
2. EMAIL COLLECTION: When customer needs account-specific help but no email provided
3. ZENDESK LOOKUP: Only when customer asks for specific data that requires Zendesk

AVAILABLE FUNCTIONS (use only when needed):
1. get_customer_info(email) - Get customer details by email
2. search_customer_tickets(email, search_type) - Search for tickets/appointments/orders
3. create_support_ticket(email, subject, description, priority) - Create support tickets (will confirm email for new customers)
4. confirm_email_and_create_ticket(email, subject, description, priority) - Confirm email and create ticket for new customers
5. add_comment_to_ticket(ticket_id, comment, public) - Add comments to tickets
6. escalate_to_billing(email, reason, priority) - Escalate a customer request to the billing department

SMART RULES:
- GREETINGS: "Hello", "Hi", "How are you" → Direct response
- GENERAL HELP: "What can you help with?", "How does this work?" → Direct response
- ACCOUNT REQUESTS without email: "I need help with my account" → Ask for email
- SPECIFIC DATA with email: "How much did I pay?", "When is my appointment?" → Use Zendesk functions
- REFUND REQUESTS: Always check customer account first using get_customer_info()
- If customer found → Use escalate_to_billing() to escalate their refund request
- If customer not found → Ask them to provide correct email
- ESCALATION REQUESTS: Only escalate AFTER confirming customer exists

EMAIL CONFIRMATION FLOW:
- When create_support_ticket() is called for a new customer, the system will automatically ask for email confirmation
- If customer confirms email, the system will create the customer account and ticket
- If customer denies email, the system will ask for the correct email

CRITICAL RULE: NEVER call create_support_ticket() without an email address. If no email is provided, ask for it first.

REFUND REQUEST FLOW:
- When customer asks for refund → Use get_customer_info() to check if they exist
- If customer found → Use escalate_to_billing() to escalate their refund request
- If customer not found → Ask them to provide correct email
- Maintain context: Don't ask for email again if already provided in conversation

EXAMPLES:
- "Hello" → Direct response
- "What can you help me with?" → Direct response  
- "I need help with my account" → "I'll need your email address to help you with that. Could you please provide your email address?"
- "My email is jeff@gmail.com" → Use get_customer_info()
- "When is my appointment?" (with email) → Use search_customer_tickets()
- "How much did I pay?" (with email) → Use search_customer_tickets()
- "I need help with a refund" (with email) → Use get_customer_info() first, then escalate_to_billing() if found
- "I need help with a refund" (no email) → "I'll need your email address to help you with that. Could you please provide your email address?"
- "Can you escalate my refund request?" (with email) → Use get_customer_info() first, then escalate_to_billing() if found

CONTEXT AWARENESS:
- If customer data is available, use it to provide personalized responses
- If customer asks about escalation/billing, use escalate_to_billing() function
- Maintain conversation context and don't ask for email if already provided
- For refund requests, ALWAYS check customer account first before escalating
- If customer not found, ask for correct email while maintaining context

Current conversation context:"""

    if conversation_history:
        prompt += "\n\nRecent conversation:\n"
        for msg in recent_messages:
            role = "Customer" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"
    
    if customer_data:
        prompt += f"\n\nCustomer data available:\n"
        prompt += f"- Name: {customer_data.get('name', 'Unknown')}\n"
        prompt += f"- Email: {customer_data.get('email', 'Unknown')}\n"
        prompt += f"- Phone: {customer_data.get('phone', 'Not provided')}\n"
    
    # Add customer email if available but no customer data yet
    if customer_email and not customer_data:
        prompt += f"\n\nCustomer email provided: {customer_email}\n"
    
    # Add conversation context for better flow
    if len(conversation_history) > 2:
        prompt += f"\n\nConversation context: The customer has been asking about rescheduling their appointment. "
        if customer_email:
            prompt += f"They provided their email: {customer_email}. "
        if customer_data:
            prompt += f"I have their account information. "
        # Check if we already found appointments
        recent_responses = [msg["content"] for msg in conversation_history[-4:] if msg["role"] == "assistant"]
        if any("appointment" in response.lower() and "found" in response.lower() for response in recent_responses):
            prompt += f"I already found their appointment information. "
        prompt += "Continue helping them with their request.\n"
    
    prompt += f"\n\nCustomer says: {transcript}\n\n"
    prompt += "IMPORTANT: Choose ONE response type only:\n\n"
    prompt += "OPTION 1 - Function Call: If you need to call a Zendesk function, respond EXACTLY with this JSON format:\n"
    prompt += '{"function": "function_name", "arguments": {...}}\n\n'
    prompt += "OPTION 2 - Direct Response: If providing a direct response, respond naturally WITHOUT any JSON.\n\n"
    prompt += "SPECIAL RULES:\n"
    prompt += "- If customer asks to reschedule and you have their appointment info, use create_support_ticket()\n"
    prompt += "- If customer provides email for the first time, use get_customer_info()\n"
    prompt += "- If customer asks about appointments and you have their email, use search_customer_tickets()\n"
    prompt += "- DO NOT ask for email if already provided\n"
    prompt += "- DO NOT ask for appointment details if already found\n\n"
    prompt += "DO NOT mix both types. Choose either function call OR direct response, not both.\n\n"
    prompt += "Your response:"
    
    return prompt

async def process_with_function_calling(transcript: str) -> str:
    """Process customer message with intelligent function calling and conversation flow"""
    global customer_email, customer_data, conversation_history, pending_ticket_request, email_confirmation_pending, last_intent, last_intent_details
    
    # Extract email if present
    extracted_email = extract_email_from_text(transcript)
    if extracted_email:
        customer_email = extracted_email
        print(f"📧 Extracted email: {extracted_email}")
    else:
        print(f"📧 No email found in transcript: '{transcript}'")
    
    # Track last intent based on conversation, not just keywords
    transcript_lower = transcript.lower().strip()
    if any(word in transcript_lower for word in ["refund", "money back", "return", "cancel"]):
        last_intent = "refund"
        last_intent_details = transcript
    elif any(word in transcript_lower for word in ["escalate", "billing"]):
        last_intent = "escalate"
        last_intent_details = transcript
    elif any(word in transcript_lower for word in ["appointment", "order", "ticket", "support"]):
        last_intent = "support"
        last_intent_details = transcript
    # else: keep previous last_intent
    
    # Handle email confirmation flow
    if email_confirmation_pending and pending_ticket_request:
        transcript_lower = transcript.lower().strip()
        confirmation_keywords = ["yes", "correct", "right", "that's right", "that is correct", "confirm", "okay", "ok"]
        denial_keywords = ["no", "wrong", "incorrect", "that's wrong", "that is wrong", "not correct", "different"]
        
        if any(keyword in transcript_lower for keyword in confirmation_keywords):
            # Email confirmed - create customer and ticket
            print(f"✅ Email confirmed: {customer_email}")
            email_confirmation_pending = False
            
            # Use the confirm_email_and_create_ticket function
            result = await execute_zendesk_function("confirm_email_and_create_ticket", pending_ticket_request)
            pending_ticket_request = None
            
            if result.get("success"):
                response_text = result.get("message", "Your request has been submitted successfully.")
                conversation_history.append({"role": "user", "content": transcript})
                conversation_history.append({"role": "assistant", "content": response_text})
                return response_text
            else:
                response_text = f"I'm sorry, but I encountered an issue: {result.get('error', 'Unknown error')}. Please try again or contact support directly."
                conversation_history.append({"role": "user", "content": transcript})
                conversation_history.append({"role": "assistant", "content": response_text})
                return response_text
                
        elif any(keyword in transcript_lower for keyword in denial_keywords):
            # Email denied - ask for correct email
            print(f"❌ Email denied: {customer_email}")
            email_confirmation_pending = False
            pending_ticket_request = None
            customer_email = None
            
            response_text = "I understand. Could you please provide the correct email address?"
            conversation_history.append({"role": "user", "content": transcript})
            conversation_history.append({"role": "assistant", "content": response_text})
            return response_text
        else:
            # Unclear response - ask for clarification
            response_text = "I didn't catch that. Could you please confirm if that email address is correct? Just say 'yes' or 'no'."
            conversation_history.append({"role": "user", "content": transcript})
            conversation_history.append({"role": "assistant", "content": response_text})
            return response_text
    
    # Check if this is a simple greeting or general question (no function needed)
    simple_greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    general_questions = ["how are you", "what can you help with", "what can you do", "how does this work"]
    
    if any(greeting in transcript_lower for greeting in simple_greetings):
        if customer_data:
            response_text = f"Hello {customer_data.get('name', 'there')}! I'm here to help. What would you like to know about your account or appointments?"
            conversation_history.append({"role": "user", "content": transcript})
            conversation_history.append({"role": "assistant", "content": response_text})
            return response_text
        else:
            response_text = "Hello! I'm your customer service assistant. I can help you with account information, appointments, orders, and support requests. How can I assist you today?"
            conversation_history.append({"role": "user", "content": transcript})
            conversation_history.append({"role": "assistant", "content": response_text})
            return response_text
    
    if any(question in transcript_lower for question in general_questions):
        if customer_data:
            response_text = f"Since I have your account information, I can help you with your appointments, orders, payments, and support requests. What would you like to know?"
        else:
            response_text = "I can help you with your account information, check appointments and orders, create support tickets, and answer questions about your services. What would you like to know?"
        conversation_history.append({"role": "user", "content": transcript})
        conversation_history.append({"role": "assistant", "content": response_text})
        return response_text
    
    # Check if customer is asking for account-specific help but no email provided
    account_keywords = ["my account", "my appointment", "my order", "my ticket", "my payment", "how much did i pay", "when is my", "escalate", "billing"]
    needs_email = any(keyword in transcript_lower for keyword in account_keywords) and not customer_data and not customer_email
    
    if needs_email:
        response_text = "I'll need your email address to help you with that. Could you please provide your email address?"
        conversation_history.append({"role": "user", "content": transcript})
        conversation_history.append({"role": "assistant", "content": response_text})
        return response_text
    
    # Build function calling prompt for more complex requests
    prompt = build_function_calling_prompt(transcript, conversation_history, customer_data)
    
    print(f"📧 Current customer_email: {customer_email}")
    print(f"📧 Current customer_data: {customer_data}")
    print(f"🧠 Last intent: {last_intent}")
    
    try:
        # Get LLM response
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        print(f"🤖 LLM Raw Response: {response_text}")
        
        # Check if response contains a function call (either standalone or mixed)
        function_call_found = False
        function_call_json = None
        
        # First, try to find a complete JSON object in the response
        try:
            # Look for JSON objects in the response by finding the last complete JSON
            # Split by newlines and look for JSON objects
            lines = response_text.split('\n')
            for line in reversed(lines):  # Start from the end
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        # Test if it's valid JSON
                        test_json = json.loads(line)
                        if 'function' in test_json:
                            function_call_json = line
                            function_call_found = True
                            print(f"🤖 Found function call JSON: {function_call_json}")
                            break
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON found in lines, try to extract JSON from the entire response using regex
            if not function_call_found:
                # Look for JSON pattern in the entire response
                json_pattern = r'\{[^{}]*"function"[^{}]*\}'
                json_matches = re.findall(json_pattern, response_text)
                if json_matches:
                    # Try to parse the last match
                    for match in reversed(json_matches):
                        try:
                            test_json = json.loads(match)
                            if 'function' in test_json:
                                function_call_json = match
                                function_call_found = True
                                print(f"🤖 Found function call JSON in mixed response: {function_call_json}")
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"🤖 JSON search error: {e}")
        
        if function_call_found and function_call_json:
            try:
                function_call = json.loads(function_call_json)
                function_name = function_call.get("function")
                arguments = function_call.get("arguments", {})
                
                print(f"🤖 Parsed function call - Name: {function_name}, Args: {arguments}")
                
                if function_name and arguments:
                    print(f"🤖 Calling function: {function_name} with args: {arguments}")
                    
                    # Safety check: Don't call create_support_ticket without email
                    if function_name == "create_support_ticket" and not arguments.get("email"):
                        response_text = "I'll need your email address to help you with that. Could you please provide your email address?"
                        conversation_history.append({"role": "user", "content": transcript})
                        conversation_history.append({"role": "assistant", "content": response_text})
                        return response_text
                    
                    # Handle refund/escalation/support context after account confirmation
                    if function_name == "get_customer_info" and customer_email:
                        result = await execute_zendesk_function(function_name, arguments)
                        if result.get("success"):
                            customer_data = result.get("customer")
                            # If last intent was refund or escalate, continue that flow
                            if last_intent == "refund" or last_intent == "escalate":
                                escalation_result = await execute_zendesk_function("escalate_to_billing", {
                                    "email": customer_email,
                                    "reason": last_intent_details or "Refund request",
                                    "priority": "high"
                                })
                                if escalation_result.get("success"):
                                    response_text = escalation_result.get("message")
                                else:
                                    response_text = "I've escalated your request to our billing department. They will review your case and contact you within 24 hours."
                                conversation_history.append({"role": "user", "content": transcript})
                                conversation_history.append({"role": "assistant", "content": response_text})
                                return response_text
                            else:
                                customer_info = result.get("customer")
                                response_text = f"Great! I found your account. Welcome back, {customer_info['name']}. How can I help you today?"
                                conversation_history.append({"role": "user", "content": transcript})
                                conversation_history.append({"role": "assistant", "content": response_text})
                                return response_text
                        else:
                            response_text = f"I don't see any account details using the email {customer_email}. Could you please check and provide me with the correct email address so I can help you with your request?"
                            conversation_history.append({"role": "user", "content": transcript})
                            conversation_history.append({"role": "assistant", "content": response_text})
                            return response_text
                    
                    # Handle escalation requests - check if customer exists first
                    if function_name == "escalate_to_billing":
                        if customer_email:
                            customer_check = await execute_zendesk_function("get_customer_info", {"email": customer_email})
                            if customer_check.get("success"):
                                response_text = (await execute_zendesk_function(function_name, arguments)).get('message')
                            else:
                                response_text = f"I don't see any account details using the email {customer_email}. Could you please check and provide me with the correct email address so I can escalate your request?"
                        else:
                            response_text = (await execute_zendesk_function(function_name, arguments)).get('message')
                        conversation_history.append({"role": "user", "content": transcript})
                        conversation_history.append({"role": "assistant", "content": response_text})
                        return response_text
                    
                    # Handle email confirmation for new customers
                    if function_name == "create_support_ticket" and result.get("needs_confirmation"):
                        email = result.get("email")
                        pending_ticket_request = arguments
                        email_confirmation_pending = True
                        
                        response_text = f"I don't see an account with the email {email}. Is this email address correct? Please confirm by saying 'yes' or 'no'."
                        conversation_history.append({"role": "user", "content": transcript})
                        conversation_history.append({"role": "assistant", "content": response_text})
                        return response_text
                    
                    # Default: Execute the function and respond
                    result = await execute_zendesk_function(function_name, arguments)
                    print(f"🤖 Function result: {result}")
                    
                    # Generate response based on function result
                    if result.get("success"):
                        if function_name == "get_customer_info":
                            customer_info = result.get("customer")
                            response_text = f"Great! I found your account. Welcome back, {customer_info['name']}. How can I help you today?"
                        elif function_name == "search_customer_tickets":
                            if "appointments" in result:
                                appointments = result.get("appointments", [])
                                if appointments:
                                    last_appointment = appointments[0]  # Store the first appointment for context
                                    response_text = f"I found {len(appointments)} appointment(s) for you. "
                                    for appt in appointments[:3]:  # Show first 3
                                        subject = appt.get('subject', 'No subject')
                                        description = appt.get('description', '')
                                        last_appointment = appt  # Update last_appointment for each
                                        date_match = re.search(r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})', description, re.IGNORECASE)
                                        if date_match:
                                            appointment_date = date_match.group(1)
                                            response_text += f"Your {subject} appointment is scheduled for {appointment_date}. "
                                        else:
                                            response_text += f"Appointment: {subject}. "
                                        if "paid" in description.lower():
                                            payment_match = re.search(r'(\d+)\s*USD', description)
                                            if payment_match:
                                                amount = payment_match.group(1)
                                                response_text += f"You paid {amount} USD for this service. "
                                        # Always include full description for context
                                        response_text += f"Details: {description} "
                                    response_text += "Would you like me to help you reschedule this appointment?"
                                else:
                                    response_text = "I don't see any appointments scheduled for you at the moment."
                            elif "orders" in result:
                                orders = result.get("orders", [])
                                if orders:
                                    last_order = orders[0]  # Store the first order for context
                                    response_text = f"I found {len(orders)} order(s) for you. "
                                    for order in orders[:3]:  # Show first 3
                                        subject = order.get('subject', 'No subject')
                                        description = order.get('description', '')
                                        last_order = order  # Update last_order for each
                                        response_text += f"Order #{order['id']}: {subject}. "
                                        # Parse invoice and product info from description
                                        invoice_match = re.search(r'Invoice\s*-\s*(\S+)', description, re.IGNORECASE)
                                        product_match = re.search(r'Product\s*-\s*([\w\s]+)', description, re.IGNORECASE)
                                        if invoice_match:
                                            response_text += f"Invoice: {invoice_match.group(1)}. "
                                        if product_match:
                                            response_text += f"Product: {product_match.group(1)}. "
                                        # Always include full description for context
                                        response_text += f"Details: {description} "
                                else:
                                    response_text = "I don't see any orders in your account at the moment."
                            else:
                                tickets = result.get("tickets", [])
                                if tickets:
                                    last_ticket = tickets[0]  # Store the first ticket for context
                                    response_text = f"I found {len(tickets)} ticket(s) for you. "
                                    for ticket in tickets[:3]:  # Show first 3
                                        subject = ticket.get('subject', 'No subject')
                                        description = ticket.get('description', '')
                                        last_ticket = ticket  # Update last_ticket for each
                                        response_text += f"Ticket #{ticket['id']}: {subject}. "
                                        # Always include full description for context
                                        response_text += f"Details: {description} "
                                else:
                                    response_text = "I don't see any open tickets for you at the moment."
                        elif function_name == "create_support_ticket":
                            response_text = f"Perfect! {result.get('message')} Our team will review your request and get back to you soon."
                        elif function_name == "confirm_email_and_create_ticket":
                            response_text = f"Perfect! {result.get('message')} Our team will review your request and get back to you soon."
                        elif function_name == "add_comment_to_ticket":
                            response_text = f"Great! {result.get('message')}. Your information has been added to the ticket."
                        elif function_name == "escalate_to_billing":
                            response_text = result.get('message')
                        # Follow-up: If user asks for more details, use last_ticket/last_order/last_appointment
                        elif last_intent in ["order", "ticket", "appointment"] and (last_ticket or last_order or last_appointment):
                            if last_intent == "order" and last_order:
                                subject = last_order.get('subject', 'No subject')
                                description = last_order.get('description', '')
                                response_text = f"Order Details: {subject}. {description} "
                                invoice_match = re.search(r'Invoice\s*-\s*(\S+)', description, re.IGNORECASE)
                                product_match = re.search(r'Product\s*-\s*([\w\s]+)', description, re.IGNORECASE)
                                if invoice_match:
                                    response_text += f"Invoice: {invoice_match.group(1)}. "
                                if product_match:
                                    response_text += f"Product: {product_match.group(1)}. "
                            elif last_intent == "ticket" and last_ticket:
                                subject = last_ticket.get('subject', 'No subject')
                                description = last_ticket.get('description', '')
                                response_text = f"Ticket Details: {subject}. {description} "
                            elif last_intent == "appointment" and last_appointment:
                                subject = last_appointment.get('subject', 'No subject')
                                description = last_appointment.get('description', '')
                                response_text = f"Appointment Details: {subject}. {description} "
                        else:
                            error_msg = result.get("error", "Unknown error occurred")
                            response_text = f"I'm sorry, but I encountered an issue: {error_msg}. Please try again or contact support directly."
                    else:
                        error_msg = result.get("error", "Unknown error occurred")
                        response_text = f"I'm sorry, but I encountered an issue: {error_msg}. Please try again or contact support directly."
                    conversation_history.append({"role": "user", "content": transcript})
                    conversation_history.append({"role": "assistant", "content": response_text})
                    return response_text
                else:
                    print("🤖 Using direct response (no function call)")
                    conversation_history.append({"role": "user", "content": transcript})
                    conversation_history.append({"role": "assistant", "content": response_text})
                    return response_text
            except json.JSONDecodeError as e:
                print(f"🤖 JSON parsing error: {e}")
                print("🤖 Using direct response (JSON parsing failed)")
                conversation_history.append({"role": "user", "content": transcript})
                conversation_history.append({"role": "assistant", "content": response_text})
                return response_text
        else:
            print("🤖 Using direct response (no function call)")
            conversation_history.append({"role": "user", "content": transcript})
            conversation_history.append({"role": "assistant", "content": response_text})
            return response_text
    except Exception as e:
        print(f"🤖 LLM processing error: {e}")
        response_text = "I'm sorry, I'm having trouble processing your request right now. Please try again."
        conversation_history.append({"role": "user", "content": transcript})
        conversation_history.append({"role": "assistant", "content": response_text})
        return response_text

def process_with_gemini(transcript):
    """Send transcript to Gemini with intelligent function calling."""
    global conversation_history, conversation_active, conversation_topic
    
    try:
        # Add user input to conversation history
        conversation_history.append({"role": "user", "content": transcript})
        
        # Detect conversation topic if not set
        if not conversation_topic and len(conversation_history) == 1:
            topic_prompt = f"""Analyze this user input and identify the main topic or intent: "{transcript}"

Return only the topic in 1-3 words (e.g., "refund", "email help", "greeting", "technical support")."""
            try:
                topic_response = gemini_model.generate_content(topic_prompt)
                conversation_topic = topic_response.text.strip().lower()
                print(f"🎯 Detected conversation topic: {conversation_topic}")
            except:
                conversation_topic = "general"
        
        # Process with function calling
        response_text = asyncio.run(process_with_function_calling(transcript))
        
        # Add assistant response to conversation history
        conversation_history.append({"role": "assistant", "content": response_text})
        
        # Keep conversation history manageable (last 6 exchanges)
        if len(conversation_history) > 12:
            conversation_history = conversation_history[-12:]
        
        return response_text
        
    except Exception as e:
        print(f"Error in process_with_gemini: {e}")
        error_msg = "I'm sorry, I didn't catch that. Could you please repeat?"
        return error_msg

def speak_text_elevenlabs(text):
    """Send text to ElevenLabs TTS and play the audio."""
    global is_speaking
    
    try:
        # Clean text: remove special characters, limit length
        text = text.strip().replace('\n', ' ').replace('\r', '')
        if len(text) > 500:
            text = text[:500] + "..."
        
        if not text:  # Don't send empty text
            return
        
        # Wait for any current TTS to finish
        while is_speaking.is_set():
            time.sleep(0.1)
        
        # Clear audio queue to prevent mixing
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break
        
        print(f"Sending TTS to ElevenLabs: {text}")
        is_speaking.set()
        
        # ElevenLabs TTS API call
        url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"  # Default voice ID
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            # Convert audio to the format we need
            audio_data = response.content
            
            # Convert MP3 to WAV format that sounddevice can play
            try:
                from pydub import AudioSegment
                from pydub.playback import play
                
                # Load MP3 and convert to WAV
                audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
                
                # Convert to 16kHz mono for compatibility
                audio = audio.set_frame_rate(16000).set_channels(1)
                
                # Convert to numpy array
                samples = np.array(audio.get_array_of_samples())
                
                # Normalize to int16
                if audio.sample_width == 2:
                    samples = samples.astype(np.int16)
                else:
                    samples = (samples * 32767).astype(np.int16)
                
                # Add to audio queue in chunks
                chunk_size = 1024
                for i in range(0, len(samples), chunk_size):
                    chunk = samples[i:i + chunk_size]
                    audio_queue.put(chunk.tobytes())
                
                print("ElevenLabs TTS audio queued successfully")
                
            except ImportError:
                print("pydub not installed. Please install it with: pip install pydub")
                # Fallback: save to file and play
                with open("temp_tts.mp3", "wb") as f:
                    f.write(audio_data)
                print("Audio saved to temp_tts.mp3 - please play manually")
                
        else:
            print(f"ElevenLabs TTS error: {response.status_code} - {response.text}")
            is_speaking.clear()
            
    except Exception as e:
        print(f"ElevenLabs TTS error: {e}")
        is_speaking.clear()

def audio_playback_thread():
    """Dedicated thread for real-time audio streaming."""
    audio_buffer = np.array([], dtype=np.int16)
    stream = None
    
    def audio_callback(outdata, frames, time, status):
        nonlocal audio_buffer
        if status:
            print(f"Audio callback status: {status}")
        
        # Fill output buffer with available audio data
        if len(audio_buffer) >= frames:
            outdata[:, 0] = audio_buffer[:frames] / 32768.0  # Convert to float32 and normalize
            audio_buffer = audio_buffer[frames:]
        else:
            # Not enough data, pad with zeros (silence)
            if len(audio_buffer) > 0:
                outdata[:len(audio_buffer), 0] = audio_buffer / 32768.0
                outdata[len(audio_buffer):, 0] = 0
                audio_buffer = np.array([], dtype=np.int16)
            else:
                outdata.fill(0)
    
    try:
        # Create output stream for real-time playback
        stream = sd.OutputStream(
            samplerate=16000,
            channels=1,
            dtype='float32',
            callback=audio_callback,
            blocksize=1024,  # Small buffer for low latency
            latency='low'
        )
        stream.start()
        
        while True:
            try:
                audio_data = audio_queue.get(timeout=1)
                if audio_data is None:  # Shutdown signal
                    break
                
                # Convert bytes to numpy array and append to buffer
                new_audio = np.frombuffer(audio_data, dtype=np.int16)
                audio_buffer = np.concatenate([audio_buffer, new_audio])
                
                # Set speaking flag when we start receiving audio
                if not is_speaking.is_set() and len(audio_buffer) > 0:
                    is_speaking.set()
                    print("Started speaking...")
                
                # Clear speaking flag when buffer is getting low and queue is empty
                if is_speaking.is_set() and len(audio_buffer) < 1024:
                    if audio_queue.empty():
                        # Wait a bit more to ensure all audio is processed
                        time.sleep(0.2)
                        if audio_queue.empty() and len(audio_buffer) < 512:
                            is_speaking.clear()
                            print("Finished speaking.")
                
                audio_queue.task_done()
                
            except queue.Empty:
                # No new audio data, check if we should clear speaking flag
                if is_speaking.is_set() and len(audio_buffer) < 512:
                    # Wait a bit longer to ensure all audio is processed
                    time.sleep(0.3)
                    if len(audio_buffer) < 256:
                        is_speaking.clear()
                        print("Finished speaking (timeout).")
                continue
                
    except Exception as e:
        print(f"Audio playback error: {e}")
    finally:
        if stream:
            stream.stop()
            stream.close()
        is_speaking.clear()

def on_stt_message(ws, message):
    """Handle Deepgram STT WebSocket messages with utterance-based processing."""
    global transcript_buffer
    try:
        data = json.loads(message)
        
        if data.get('type') == 'UtteranceEnd':
            # Process the complete utterance when we get the end signal
            if transcript_buffer:
                full_transcript = " ".join(transcript_buffer).strip()
                transcript_buffer.clear()  # Clear buffer for next utterance
                
                if full_transcript:
                    # Skip processing if we're currently speaking (avoid feedback)
                    if is_speaking.is_set():
                        print(f"Ignoring transcript while speaking: {full_transcript}")
                        return
                    
                    print(f"🎯 Utterance Complete: '{full_transcript}'")
                    
                    # Get only the refined response from Gemini
                    response_text = process_with_gemini(full_transcript)
                    print(f"🤖 Response: {response_text}")
                    
                    # Save to file with clean format
                    with open("transcripts.txt", "a", encoding="utf-8") as f:
                        f.write(f"User: {full_transcript}\nAssistant: {response_text}\n\n")
                    
                    # Send to ElevenLabs TTS
                    if response_text:
                        speak_text_elevenlabs(response_text)
            else:
                print("⚠️  UtteranceEnd received but no transcript in buffer")
                
        elif 'channel' in data and 'alternatives' in data['channel']:
            transcript = data['channel']['alternatives'][0].get('transcript', '')
            is_final = data.get('is_final', False)
            is_interim = data.get('is_interim', False)
            
            if transcript:
                if is_final:
                    # Add final transcript segments to buffer
                    transcript_buffer.append(transcript)
                    print(f"📝 Buffered final segment: '{transcript}'")
                    
                    # Optional: Log speech final for debugging
                    if data.get('speech_final', False):
                        print(f"🔚 Speech final segment: '{transcript}'")
                elif is_interim:
                    # Log interim results for debugging (optional)
                    print(f"🔄 Interim: '{transcript}'")
                    
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON response from Deepgram STT")
    except Exception as e:
        print(f"❌ STT message handling error: {e}")
        import traceback
        traceback.print_exc()

def on_stt_error(ws, error):
    """Handle Deepgram STT WebSocket errors."""
    print(f"STT Error: {error}")

def on_stt_close(ws, close_status_code, close_msg):
    """Handle Deepgram STT WebSocket closure."""
    print(f"STT Connection closed (Status: {close_status_code}, Message: {close_msg})")
    # Shutdown audio playback thread
    audio_queue.put(None)

def on_stt_open(ws):
    """Start audio streaming on Deepgram STT WebSocket open."""
    def run(*args):
        last_overflow_time = 0
        
        def callback(indata, frames, time_info, status):
            nonlocal last_overflow_time
            if status:
                current_time = time.time()
                if status.input_overflow and (current_time - last_overflow_time > 1):
                    print("Warning: Audio input overflow detected")
                    last_overflow_time = current_time
            
            # Send audio data to Deepgram
            if is_speaking.is_set():
                # Send silence during TTS to prevent connection timeout
                silence = np.zeros(frames, dtype=np.int16).tobytes()
                try:
                    ws.send(silence, opcode=websocket.ABNF.OPCODE_BINARY)
                except Exception as e:
                    print(f"Error sending silence: {e}")
            else:
                # Send actual audio when not speaking
                audio = (indata * 32767).astype(np.int16).tobytes()
                try:
                    ws.send(audio, opcode=websocket.ABNF.OPCODE_BINARY)
                except Exception as e:
                    print(f"Error sending audio: {e}")
        
        # Try to use default input device first, then fallback to device 1
        input_device = None
        try:
            # Test if device 1 works
            with sd.InputStream(device=1, samplerate=16000, channels=1, dtype='float32', blocksize=1024):
                pass
            input_device = 1
            print("Using device 1 (Nothing Ear)")
        except:
            print("Device 1 not available, using default input device")
            input_device = None
        
        with sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype='float32',
            device=input_device,
            blocksize=4096,  # Balanced blocksize for good responsiveness
            callback=callback,
            latency='low'  # Low latency for real-time processing
        ):
            print("Speak now... Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                ws.close()
                
    threading.Thread(target=run, daemon=True).start()

def stt_keepalive():
    """Send periodic keepalive messages to STT WebSocket."""
    global ws_stt
    while True:
        try:
            time.sleep(30)  # Send keepalive every 30 seconds
            with stt_lock:
                if ws_stt and hasattr(ws_stt, 'sock') and ws_stt.sock and ws_stt.sock.connected:
                    # Send a ping to keep connection alive
                    ws_stt.send("", opcode=websocket.ABNF.OPCODE_PING)
                    print("🔌 STT keepalive ping sent")
        except Exception as e:
            print(f"STT keepalive error: {e}")
            break

def extract_email_from_text(text: str) -> Optional[str]:
    """Extract email address from text"""
    print(f"🔍 Extracting email from: '{text}'")
    
    # Multiple patterns to catch different formats
    email_patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Standard email
        r'[A-Za-z0-9._%+-]+\s*at\s*[A-Za-z0-9.-]+\s*dot\s*[A-Z|a-z]{2,}',  # Spoken format
        r'[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Z|a-z]{2,}',  # With spaces
    ]
    
    for pattern in email_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            email = match.group(0)
            # Clean up the email (remove extra spaces, convert "at" and "dot")
            email = re.sub(r'\s+', '', email)  # Remove spaces
            email = re.sub(r'\bat\b', '@', email, flags=re.IGNORECASE)  # Convert "at" to @
            email = re.sub(r'\bdot\b', '.', email, flags=re.IGNORECASE)  # Convert "dot" to .
            print(f"✅ Found email: {email}")
            return email
    
    print(f"❌ No email found in text")
    return None

def detect_customer_intent(text: str) -> Dict:
    """Detect customer intent and required information"""
    text_lower = text.lower()
    
    intent = {
        "type": "general",
        "requires_email": False,
        "requires_account_access": False,
        "action_needed": None
    }
    
    # Check for account-related requests
    account_keywords = ["account", "profile", "details", "information", "change", "update", "modify"]
    if any(keyword in text_lower for keyword in account_keywords):
        intent["type"] = "account_management"
        intent["requires_email"] = True
        intent["requires_account_access"] = True
    
    # Check for order-related requests
    order_keywords = ["order", "purchase", "transaction", "payment", "invoice", "receipt"]
    if any(keyword in text_lower for keyword in order_keywords):
        intent["type"] = "order_inquiry"
        intent["requires_email"] = True
    
    # Check for meeting/appointment requests
    meeting_keywords = ["meeting", "appointment", "schedule", "booking", "call", "evisa"]
    if any(keyword in text_lower for keyword in meeting_keywords):
        intent["type"] = "meeting_management"
        intent["requires_email"] = True
    
    # Check for support requests
    support_keywords = ["help", "support", "issue", "problem", "ticket", "complaint"]
    if any(keyword in text_lower for keyword in support_keywords):
        intent["type"] = "support_request"
        intent["requires_email"] = True
        intent["action_needed"] = "create_ticket"
    
    return intent

def main():
    """Initialize STT WebSocket client with ElevenLabs TTS and Zendesk integration."""
    print("Starting Enhanced Voice Agent with Zendesk Integration")
    print("=" * 60)
    print("Features:")
    print("- ElevenLabs TTS for natural voice")
    print("- Deepgram STT for accurate transcription")
    print("- Zendesk integration for customer data")
    print("- Smart email detection and customer lookup")
    print("- Ticket creation and management")
    print("- Conversation context awareness")
    print("=" * 60)
    
    print("Available audio input devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"Device {i}: {dev['name']} (Input Channels: {dev['max_input_channels']})")
    
    # Start audio playback thread
    playback_thread = threading.Thread(target=audio_playback_thread, daemon=True)
    playback_thread.start()
    
    # Start STT keepalive thread
    keepalive_thread = threading.Thread(target=stt_keepalive, daemon=True)
    keepalive_thread.start()
    
    try:
        # Initialize STT WebSocket
        global ws_stt
        ws_stt = websocket.WebSocketApp(
            DEEPGRAM_STT_URL,
            header=[f"Authorization: Token {DEEPGRAM_API_KEY}"],
            on_open=on_stt_open,
            on_message=on_stt_message,
            on_error=on_stt_error,
            on_close=on_stt_close,
        )
        ws_stt.run_forever()
        
    except Exception as e:
        print(f"WebSocket initialization error: {e}")
    finally:
        # Cleanup
        audio_queue.put(None)  # Signal audio thread to stop
        with stt_lock:
            if ws_stt and hasattr(ws_stt, 'sock') and ws_stt.sock:
                ws_stt.close()

if __name__ == "__main__":
    main() 