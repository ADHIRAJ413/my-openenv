"""
Email Generator — Synthetic email dataset generation for the Email Triage environment.

Generates realistic emails with deterministic seeding for reproducibility.
Each email has ground truth labels for category, priority, department, and response needs.
"""

import random
import string
from datetime import datetime, timedelta
from typing import Optional
from models import EmailItem, EmailGroundTruth


# ── Email Templates ───────────────────────────────────────────────

SENDER_POOLS = {
    "billing": [
        ("john.smith@acmecorp.com", "John Smith"),
        ("sarah.jones@widgets.io", "Sarah Jones"),
        ("finance@bigretail.com", "BigRetail Finance"),
        ("ap@techstartup.co", "Accounts Payable"),
        ("peter.wong@enterprise.net", "Peter Wong"),
    ],
    "technical": [
        ("devops@cloudhost.io", "CloudHost DevOps"),
        ("alice.chen@devteam.com", "Alice Chen"),
        ("support@saasprovider.com", "SaaS Support"),
        ("k.patel@eng.startup.io", "Kunal Patel"),
        ("sysadmin@infra.net", "Sysadmin Team"),
    ],
    "sales": [
        ("mike@salesforce.example.com", "Mike Richards"),
        ("partnerships@vendor.co", "Vendor Partnerships"),
        ("demo@newproduct.io", "NewProduct Demo Team"),
        ("lisa.martin@solutions.com", "Lisa Martin"),
        ("bdteam@growthhq.io", "Growth HQ BD"),
    ],
    "hr": [
        ("hr@company.com", "HR Department"),
        ("benefits@company.com", "Benefits Team"),
        ("recruiting@company.com", "Recruiting"),
        ("ceo@company.com", "CEO Office"),
        ("compliance@company.com", "Compliance Team"),
    ],
    "spam": [
        ("winner@lottery-intl.xyz", "International Lottery"),
        ("noreply@deals4u.biz", "Amazing Deals"),
        ("prince@royalfund.ng", "Royal Prince"),
        ("free@crypto-earn.xyz", "Crypto Earnings"),
        ("admin@acc0unt-verify.com", "Account Security"),
    ],
    "phishing": [
        ("security@paypa1.com", "PayPal Security"),
        ("admin@micr0soft-support.com", "Microsoft Support"),
        ("noreply@bank-0f-america.com", "Bank of America"),
        ("it-support@company-portal.net", "IT Support"),
        ("ceo@c0mpany.com", "CEO Urgent"),
    ],
}

BILLING_TEMPLATES = [
    {
        "subject": "Invoice #{inv_num} — Payment Overdue",
        "body": "Dear Team,\n\nThis is a reminder that invoice #{inv_num} for ${amount} is now {days} days overdue. Please process payment at your earliest convenience to avoid late fees.\n\nOriginal due date: {due_date}\nAmount due: ${amount}\nLate fee (if applicable): ${late_fee}\n\nPlease contact us if you have any questions about this invoice.\n\nBest regards,\n{sender_name}",
        "priority": "high",
        "requires_response": True,
        "key_points": ["acknowledge receipt", "provide payment timeline", "mention processing"],
    },
    {
        "subject": "Updated pricing for Q{quarter} {year}",
        "body": "Hi,\n\nI wanted to share our updated pricing structure for Q{quarter} {year}. Key changes include:\n\n- Enterprise tier: ${ent_price}/month (was ${old_price}/month)\n- Volume discounts now start at {vol_threshold} units\n- New annual billing option with 15% discount\n\nPlease review and let me know if you have any questions. Happy to schedule a call to walk through the changes.\n\nThanks,\n{sender_name}",
        "priority": "medium",
        "requires_response": True,
        "key_points": ["acknowledge pricing changes", "ask clarifying questions"],
    },
    {
        "subject": "Payment confirmation — Order #{order_num}",
        "body": "This is an automated confirmation that your payment of ${amount} for order #{order_num} has been successfully processed.\n\nTransaction ID: {tx_id}\nDate: {tx_date}\n\nNo action required. Please keep this for your records.\n\nThank you for your business.",
        "priority": "low",
        "requires_response": False,
        "key_points": [],
    },
    {
        "subject": "Expense report requires approval",
        "body": "Hello,\n\nI've submitted expense report #{exp_id} totaling ${amount} for the {event_name} conference. Attached are all receipts.\n\nBreakdown:\n- Travel: ${travel}\n- Hotel: ${hotel}\n- Meals: ${meals}\n- Registration: ${reg}\n\nPlease approve at your earliest convenience.\n\nThanks,\n{sender_name}",
        "priority": "medium",
        "requires_response": True,
        "key_points": ["acknowledge receipt", "approve or request changes"],
    },
]

TECHNICAL_TEMPLATES = [
    {
        "subject": "URGENT: Production server {server} is down",
        "body": "ALERT: Production server {server} is unresponsive as of {time}.\n\nSymptoms:\n- HTTP 503 on all endpoints\n- CPU usage spiked to 100% before failure\n- Last successful deploy: {last_deploy}\n\nImpact: {num_users} active users affected. Revenue impact estimated at ${rev_impact}/hour.\n\nWe need immediate assistance with:\n1. Root cause analysis\n2. Rollback to last stable build if needed\n3. Status page update\n\nPlease acknowledge ASAP.\n\n— {sender_name}",
        "priority": "critical",
        "requires_response": True,
        "key_points": ["acknowledge urgency", "provide ETA", "mention escalation"],
    },
    {
        "subject": "Bug report: {feature} not working on {platform}",
        "body": "Hi team,\n\nI've found a bug in the {feature} feature:\n\nSteps to reproduce:\n1. Navigate to {url}\n2. Click on '{button}'\n3. Enter test data and submit\n\nExpected: {expected}\nActual: {actual}\n\nBrowser: {browser}\nOS: {os}\nScreenshot attached.\n\nThis is blocking {num_users} users from completing their workflow.\n\nThanks,\n{sender_name}",
        "priority": "high",
        "requires_response": True,
        "key_points": ["acknowledge bug", "provide timeline for fix", "suggest workaround"],
    },
    {
        "subject": "Feature request: {feature_name}",
        "body": "Hello,\n\nI'd like to request a new feature: {feature_name}.\n\nUse case: {use_case}\n\nThis would help our team by {benefit}. Currently we have to {workaround} which takes approximately {time_waste} hours per week.\n\nWould it be possible to add this to the roadmap for Q{quarter}?\n\nBest,\n{sender_name}",
        "priority": "low",
        "requires_response": True,
        "key_points": ["acknowledge request", "mention roadmap review"],
    },
    {
        "subject": "Scheduled maintenance: {system} downtime",
        "body": "This is a notification that {system} will undergo scheduled maintenance on {maint_date} from {start_time} to {end_time} ({timezone}).\n\nAffected services:\n- {service1}\n- {service2}\n\nExpected downtime: {downtime} minutes\n\nNo action required. We will send an all-clear notification once maintenance is complete.",
        "priority": "low",
        "requires_response": False,
        "key_points": [],
    },
]

SALES_TEMPLATES = [
    {
        "subject": "Partnership opportunity — {company_name}",
        "body": "Hi there,\n\nI'm reaching out from {company_name}. We've been following your growth and believe there's a strong opportunity for partnership.\n\nOur platform helps companies like yours with {value_prop}. We currently serve {num_clients}+ clients including {ref_client1} and {ref_client2}.\n\nWould you be open to a 30-minute call this week to explore potential synergies?\n\nLooking forward to hearing from you.\n\nBest,\n{sender_name}",
        "priority": "medium",
        "requires_response": True,
        "key_points": ["express interest or decline", "suggest timing"],
    },
    {
        "subject": "Contract renewal — account #{acct_num}",
        "body": "Dear valued customer,\n\nYour annual contract (account #{acct_num}) is up for renewal on {renewal_date}. \n\nCurrent plan: {plan_name}\nAnnual cost: ${annual_cost}\n\nWe'd like to offer a {discount}% loyalty discount if you renew before {early_date}. We also have new features in our {new_tier} tier that might interest you.\n\nShall we schedule a review call?\n\nBest regards,\n{sender_name}",
        "priority": "high",
        "requires_response": True,
        "key_points": ["acknowledge renewal", "discuss terms or schedule call"],
    },
    {
        "subject": "RE: Quote request for {product}",
        "body": "Hi,\n\nThanks for your interest in {product}. Here's the quote you requested:\n\n- {product} ({tier} tier): ${unit_price}/unit\n- Quantity: {quantity}\n- Subtotal: ${subtotal}\n- Volume discount ({disc}%): -${disc_amount}\n- Total: ${total}\n\nThis quote is valid for 30 days. Shall I prepare a formal proposal?\n\nBest,\n{sender_name}",
        "priority": "medium",
        "requires_response": True,
        "key_points": ["review pricing", "confirm or negotiate"],
    },
]

HR_TEMPLATES = [
    {
        "subject": "Annual performance review — Due {due_date}",
        "body": "Dear {employee_name},\n\nThis is a reminder that your annual performance review is due by {due_date}. Please complete the following:\n\n1. Self-assessment form (attached)\n2. Goals review for the past year\n3. Goal setting for the upcoming year\n\nYou will also need to schedule a 1:1 meeting with your manager by {meeting_deadline}.\n\nPlease reach out to HR if you have any questions.\n\nBest,\nHR Department",
        "priority": "medium",
        "requires_response": True,
        "key_points": ["acknowledge deadline", "confirm submission plan"],
    },
    {
        "subject": "New company policy: {policy_name}",
        "body": "All Staff,\n\nWe are implementing a new policy regarding {policy_name}, effective {effective_date}.\n\nKey changes:\n- {change1}\n- {change2}\n- {change3}\n\nPlease review the full policy document (attached) and acknowledge receipt by {ack_date}.\n\nQuestions? Contact HR at hr@company.com.\n\nBest regards,\nCompliance Team",
        "priority": "medium",
        "requires_response": True,
        "key_points": ["acknowledge receipt", "confirm review"],
    },
    {
        "subject": "Welcome aboard: {new_hire} starts {start_date}",
        "body": "Team,\n\nPlease welcome {new_hire}, who will be joining as {role} on {start_date}. {pronoun} will be reporting to {manager}.\n\nA few things to set up before their start date:\n- Desk/workspace assignment\n- Equipment (laptop, monitor, etc.)\n- Account provisioning\n\nPlease make sure everything is ready by {prep_date}.\n\nThanks,\nRecruiting Team",
        "priority": "medium",
        "requires_response": False,
        "key_points": [],
    },
]

SPAM_TEMPLATES = [
    {
        "subject": "🎉 YOU WON $1,000,000!!! Claim NOW!!!",
        "body": "CONGRATULATIONS!!!\n\nYou have been selected as the WINNER of our international sweepstakes! You have won $1,000,000 USD!!!\n\nTo claim your prize, simply reply with:\n- Full name\n- Bank account number\n- Social security number\n\nACT NOW! This offer expires in 24 HOURS!\n\nSincerely,\nInternational Lottery Commission",
        "priority": "low",
        "requires_response": False,
        "key_points": [],
    },
    {
        "subject": "Limited time: 90% OFF Premium Watches",
        "body": "EXCLUSIVE DEAL - TODAY ONLY!\n\nGet premium designer watches at 90% OFF retail price!\n\n🔥 Rolex Submariner - $49.99 (was $8,000)\n🔥 Omega Seamaster - $39.99 (was $5,000)\n🔥 Tag Heuer Monaco - $29.99 (was $3,500)\n\nFREE SHIPPING on all orders!\n\nClick here to order: http://t0tally-leg1t-watches.biz/deal\n\nUnsubscribe: reply STOP (just kidding, you can't)",
        "priority": "low",
        "requires_response": False,
        "key_points": [],
    },
    {
        "subject": "Make $5000/day working from home",
        "body": "Hi friend,\n\nAre you tired of your 9-5 job? What if I told you that you could make $5000 PER DAY from your couch?\n\nMy PROVEN system has helped 10,000+ people achieve financial freedom.\n\nNo experience needed!\nNo investment required (just $299 for the starter kit)!\n\nDon't miss this once-in-a-lifetime opportunity!\n\nReply NOW to get started!\n\nTo your success,\nRich McMoneyface",
        "priority": "low",
        "requires_response": False,
        "key_points": [],
    },
]

PHISHING_TEMPLATES = [
    {
        "subject": "⚠️ Urgent: Your account has been compromised",
        "body": "Dear valued customer,\n\nWe have detected suspicious activity on your account. Your account has been temporarily limited.\n\nTo restore full access, please verify your identity by clicking the link below:\n\nhttp://paypa1-secure-verify.com/login\n\nYou must verify within 24 hours or your account will be permanently suspended.\n\nThank you for your prompt attention to this matter.\n\nPayPal Security Team\n\nNote: This email was sent from a verified PayPal address. Do not ignore.",
        "priority": "low",
        "requires_response": False,
        "key_points": [],
        "is_phishing": True,
    },
    {
        "subject": "Action required: Password reset — IT Helpdesk",
        "body": "Hello,\n\nDue to a recent security upgrade, all employees must reset their passwords immediately.\n\nPlease click the following link to reset your password:\nhttp://company-portal-reset.net/password\n\nYou will need to enter your:\n- Current username\n- Current password\n- New password\n\nThis must be completed by end of day today.\n\nThank you,\nIT Support Team",
        "priority": "low",
        "requires_response": False,
        "key_points": [],
        "is_phishing": True,
    },
    {
        "subject": "RE: Wire transfer approval needed ASAP",
        "body": "Hi,\n\nI need you to process an urgent wire transfer of $45,000 to the following account:\n\nBank: First National\nAccount: 8847291034\nRouting: 021000021\nBeneficiary: Global Ventures LLC\n\nThis is for the deal we discussed last week. Please process immediately and confirm.\n\nI'm in a meeting so email is best — don't call.\n\nThanks,\nCEO",
        "priority": "low",
        "requires_response": False,
        "key_points": [],
        "is_phishing": True,
    },
]


def _random_id(prefix: str, length: int = 6) -> str:
    return f"{prefix}-{''.join(random.choices(string.digits, k=length))}"


def _fill_template(template_body: str, sender_name: str, rng: random.Random) -> str:
    """Fill template placeholders with random but realistic values."""
    now = datetime(2026, 3, 15, 9, 0, 0)

    replacements = {
        "sender_name": sender_name,
        "inv_num": str(rng.randint(10000, 99999)),
        "amount": f"{rng.randint(500, 50000):,}",
        "days": str(rng.choice([7, 14, 30, 45])),
        "due_date": (now - timedelta(days=rng.randint(7, 45))).strftime("%B %d, %Y"),
        "late_fee": f"{rng.randint(25, 500):,}",
        "quarter": str(rng.randint(1, 4)),
        "year": "2026",
        "ent_price": f"{rng.randint(500, 5000):,}",
        "old_price": f"{rng.randint(400, 4500):,}",
        "vol_threshold": str(rng.choice([50, 100, 250, 500])),
        "order_num": str(rng.randint(100000, 999999)),
        "tx_id": _random_id("TXN", 10),
        "tx_date": now.strftime("%B %d, %Y"),
        "exp_id": _random_id("EXP"),
        "event_name": rng.choice(["AWS re:Invent", "PyCon", "KubeCon", "React Summit"]),
        "travel": f"{rng.randint(200, 2000):,}",
        "hotel": f"{rng.randint(300, 1500):,}",
        "meals": f"{rng.randint(100, 500):,}",
        "reg": f"{rng.randint(200, 1000):,}",
        "server": rng.choice(["prod-api-01", "web-frontend-03", "db-primary", "worker-queue-02"]),
        "time": (now - timedelta(minutes=rng.randint(5, 60))).strftime("%H:%M UTC"),
        "last_deploy": (now - timedelta(hours=rng.randint(2, 48))).strftime("%Y-%m-%d %H:%M"),
        "num_users": f"{rng.randint(100, 50000):,}",
        "rev_impact": f"{rng.randint(500, 10000):,}",
        "feature": rng.choice(["search", "checkout", "dashboard", "export", "authentication"]),
        "platform": rng.choice(["Chrome/Mac", "Safari/iOS", "Firefox/Windows", "Edge/Windows"]),
        "url": "/dashboard/" + rng.choice(["settings", "reports", "analytics", "users"]),
        "button": rng.choice(["Submit", "Save", "Export", "Filter", "Apply"]),
        "expected": "Form submits successfully and shows confirmation",
        "actual": rng.choice(["Page shows 500 error", "Spinner never stops", "Data not saved", "Blank screen"]),
        "browser": rng.choice(["Chrome 120", "Firefox 121", "Safari 17", "Edge 120"]),
        "os": rng.choice(["macOS 14.2", "Windows 11", "Ubuntu 22.04", "iOS 17"]),
        "feature_name": rng.choice(["bulk export", "dark mode", "API webhooks", "SSO integration", "audit log"]),
        "use_case": "streamline our daily workflow",
        "benefit": "saving approximately 10 hours per week across the team",
        "workaround": "manually export data and process it in spreadsheets",
        "time_waste": str(rng.randint(3, 15)),
        "system": rng.choice(["CI/CD pipeline", "email servers", "authentication system", "CDN"]),
        "maint_date": (now + timedelta(days=rng.randint(3, 14))).strftime("%B %d, %Y"),
        "start_time": rng.choice(["02:00", "03:00", "04:00"]),
        "end_time": rng.choice(["04:00", "05:00", "06:00"]),
        "timezone": "UTC",
        "service1": rng.choice(["API Gateway", "User Authentication", "File Storage"]),
        "service2": rng.choice(["Search Index", "Background Jobs", "Notification Service"]),
        "downtime": str(rng.choice([15, 30, 45, 60, 120])),
        "company_name": rng.choice(["TechVentures", "DataFlow Inc.", "CloudNine", "InnovateCo"]),
        "value_prop": "reducing operational costs by up to 40%",
        "num_clients": str(rng.choice([500, 1000, 2000, 5000])),
        "ref_client1": rng.choice(["Stripe", "Shopify", "Atlassian", "Notion"]),
        "ref_client2": rng.choice(["Slack", "Figma", "Linear", "Vercel"]),
        "acct_num": str(rng.randint(10000, 99999)),
        "renewal_date": (now + timedelta(days=rng.randint(15, 60))).strftime("%B %d, %Y"),
        "plan_name": rng.choice(["Professional", "Enterprise", "Growth", "Starter"]),
        "annual_cost": f"{rng.randint(5000, 100000):,}",
        "discount": str(rng.choice([10, 15, 20])),
        "early_date": (now + timedelta(days=rng.randint(7, 30))).strftime("%B %d, %Y"),
        "new_tier": rng.choice(["Premium", "Enterprise Plus", "Ultimate"]),
        "product": rng.choice(["CloudSuite Pro", "DataAnalyzer", "SecureVault", "DevOps Platform"]),
        "tier": rng.choice(["Standard", "Professional", "Enterprise"]),
        "unit_price": f"{rng.randint(50, 500):,}",
        "quantity": str(rng.choice([10, 25, 50, 100, 250])),
        "subtotal": f"{rng.randint(1000, 50000):,}",
        "disc": str(rng.choice([5, 10, 15, 20])),
        "disc_amount": f"{rng.randint(100, 5000):,}",
        "total": f"{rng.randint(900, 45000):,}",
        "employee_name": rng.choice(["Team Member", "Employee"]),
        "meeting_deadline": (now + timedelta(days=rng.randint(5, 14))).strftime("%B %d, %Y"),
        "policy_name": rng.choice(["Remote Work", "Data Security", "Travel Expenses", "PTO Accrual"]),
        "effective_date": (now + timedelta(days=rng.randint(14, 30))).strftime("%B %d, %Y"),
        "change1": "Updated guidelines for approval workflows",
        "change2": "New documentation requirements",
        "change3": "Revised escalation procedures",
        "ack_date": (now + timedelta(days=rng.randint(5, 10))).strftime("%B %d, %Y"),
        "new_hire": rng.choice(["Alex Rivera", "Jordan Kim", "Sam Patel", "Casey Nguyen"]),
        "role": rng.choice(["Senior Engineer", "Product Manager", "Data Analyst", "Designer"]),
        "start_date": (now + timedelta(days=rng.randint(7, 30))).strftime("%B %d, %Y"),
        "pronoun": rng.choice(["They", "She", "He"]),
        "manager": rng.choice(["David Lee", "Maria Garcia", "James Wilson", "Emily Brown"]),
        "prep_date": (now + timedelta(days=rng.randint(5, 25))).strftime("%B %d, %Y"),
    }

    result = template_body
    for key, value in replacements.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def _get_category_label(cat: str) -> str:
    """Map internal category to the label agents should use."""
    if cat == "phishing":
        return "spam"  # Agent should categorize phishing as spam
    return cat


def _get_department(cat: str) -> str:
    """Map category to department for routing."""
    mapping = {
        "billing": "billing_dept",
        "technical": "engineering",
        "sales": "sales_team",
        "hr": "hr_dept",
        "spam": "security",
        "phishing": "security",
    }
    return mapping.get(cat, "general")


class EmailGenerator:
    """Generates synthetic email datasets with ground truth labels."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._templates = {
            "billing": BILLING_TEMPLATES,
            "technical": TECHNICAL_TEMPLATES,
            "sales": SALES_TEMPLATES,
            "hr": HR_TEMPLATES,
            "spam": SPAM_TEMPLATES,
            "phishing": PHISHING_TEMPLATES,
        }

    def generate_batch(
        self,
        count: int,
        categories: Optional[list[str]] = None,
        difficulty: str = "easy",
    ) -> tuple[list[EmailItem], list[EmailGroundTruth]]:
        """
        Generate a batch of emails with ground truth.

        Args:
            count: Number of emails to generate
            categories: Which categories to include (default: all)
            difficulty: "easy", "medium", or "hard"

        Returns:
            Tuple of (emails, ground_truths)
        """
        if categories is None:
            if difficulty == "easy":
                categories = ["billing", "technical", "sales", "hr", "spam"]
            elif difficulty == "medium":
                categories = ["billing", "technical", "sales", "hr", "spam", "phishing"]
            else:
                categories = ["billing", "technical", "sales", "hr", "spam", "phishing"]

        emails: list[EmailItem] = []
        ground_truths: list[EmailGroundTruth] = []

        base_time = datetime(2026, 3, 15, 8, 0, 0)

        for i in range(count):
            cat = self.rng.choice(categories)
            templates = self._templates[cat]
            template = self.rng.choice(templates)
            senders = SENDER_POOLS[cat]
            sender_email, sender_name = self.rng.choice(senders)

            email_id = f"email-{i+1:04d}"
            timestamp = (base_time + timedelta(minutes=self.rng.randint(1, 1440) * (i + 1))).isoformat()

            subject = _fill_template(template["subject"], sender_name, self.rng)
            body = _fill_template(template["body"], sender_name, self.rng)

            # Add threading for medium/hard difficulty
            reply_to = None
            thread_length = 1
            if difficulty in ("medium", "hard") and i > 0 and self.rng.random() < 0.25:
                reply_to = f"email-{self.rng.randint(1, i):04d}"
                thread_length = self.rng.randint(2, 5)

            has_attachment = self.rng.random() < 0.3

            email = EmailItem(
                email_id=email_id,
                sender=sender_email,
                sender_name=sender_name,
                subject=subject,
                body=body,
                timestamp=timestamp,
                has_attachment=has_attachment,
                reply_to=reply_to,
                thread_length=thread_length,
                is_read=False,
            )

            is_spam = cat in ("spam", "phishing")
            is_phishing = cat == "phishing"

            ground_truth = EmailGroundTruth(
                email_id=email_id,
                category=_get_category_label(cat),
                priority=template.get("priority", "medium"),
                department=_get_department(cat),
                requires_response=template.get("requires_response", False),
                key_response_points=template.get("key_points", []),
                is_spam=is_spam,
                is_phishing=is_phishing,
            )

            emails.append(email)
            ground_truths.append(ground_truth)

        return emails, ground_truths

    def generate_easy_set(self) -> tuple[list[EmailItem], list[EmailGroundTruth]]:
        """Generate 10 clear, unambiguous emails for the easy task."""
        return self.generate_batch(10, difficulty="easy")

    def generate_medium_set(self) -> tuple[list[EmailItem], list[EmailGroundTruth]]:
        """Generate 20 emails with moderate complexity for the medium task."""
        return self.generate_batch(20, difficulty="medium")

    def generate_hard_set(self) -> tuple[list[EmailItem], list[EmailGroundTruth]]:
        """Generate 30 emails with threads, phishing, and ambiguity for the hard task."""
        return self.generate_batch(30, difficulty="hard")
