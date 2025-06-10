import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pymongo
from datetime import datetime
import json
import faiss
import pickle
import requests
import os
from dotenv import load_dotenv, dotenv_values 
from groq import Groq
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="5G NOC Anomaly Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .anomaly-alert {
        background-color: #ffebee; 
        color: #000000;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
    }
    .normal-status {
        background-color: #000000;
        color: #000000;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
    }
    .important-features {
        background-color: #000000;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .llama-solution {
        background-color: #e3f2fd;
        color: #000000;
        border-left: 5px solid #2196f3;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .solution-section {
        background-color: #f8f9fa;
        color: #000000;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

class Config:
    """Configuration class for the application"""
    MONGODB_URI = os.getenv("MONGODB_URI") 
    DATABASE_NAME = "noc_5g_monitoring"
    NORMAL_COLLECTION = "normal_requests"
    ANOMALY_COLLECTION = "anomaly_requests"
    ALL_REQUESTS_COLLECTION = "all_requests"
    
    # File paths
    MODEL_PATH = "model.joblib"  
    SELECTED_FEATURES_PATH = "selected_feature_names.npy"
    FAISS_INDEX_PATH = r"guide_for_5g_vector_db\index.faiss"
    FAISS_PKL_PATH = r"guide_for_5g_vector_db\index.pkl"

    GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
    
    # Important features for model prediction
    IMPORTANT_FEATURES = ['data_rate_mbps', 'latency_ms', 'radio_resource_usage', 
                         'core_network_load', 'upf_load']

class DatabaseManager:
    """Handle MongoDB operations"""
    
    def __init__(self):
        try:
            self.client = pymongo.MongoClient(Config.MONGODB_URI)
            self.db = self.client[Config.DATABASE_NAME]
            st.success("✅ Connected to MongoDB")
        except Exception as e:
            st.error(f"❌ MongoDB connection failed: {str(e)}")
            self.client = None
            self.db = None
    
    def insert_request_data(self, data, collection_name):
        """Insert request data into MongoDB"""
        if self.db is None:
            return False
        try:
            data['timestamp'] = datetime.now()
            result = self.db[collection_name].insert_one(data)
            return str(result.inserted_id)
        except Exception as e:
            st.error(f"Database insertion error: {str(e)}")
            return False
    
    def get_recent_requests(self, collection_name, limit=10):
        """Get recent requests from database"""
        if self.db is None:
            return []
        try:
            return list(self.db[collection_name].find().sort("timestamp", -1).limit(limit))
        except Exception as e:
            st.error(f"Database query error: {str(e)}")
            return []

class FeatureProcessor:
    """Handle feature processing for model prediction"""
    
    def __init__(self):
        self.important_features = Config.IMPORTANT_FEATURES
    
    def extract_important_features(self, request_data):
        """Extract only the important features for model prediction"""
        feature_vector = []
        for feature in self.important_features:
            if feature in request_data:
                feature_vector.append(request_data[feature])
            else:
                # Default values if feature is missing
                default_values = {
                    'data_rate_mbps': 0.0,
                    'latency_ms': 0.0,
                    'radio_resource_usage': 0,
                    'core_network_load': 0,
                    'upf_load': 0
                }
                feature_vector.append(default_values.get(feature, 0))
        
        return np.array(feature_vector).reshape(1, -1)
    
    def validate_important_features(self, request_data):
        """Validate that all important features are present"""
        missing_features = []
        for feature in self.important_features:
            if feature not in request_data or request_data[feature] is None:
                missing_features.append(feature)
        return missing_features

class EnhancedRAGSystem:
    """Enhanced RAG system with Llama integration for NOC engineer solutions"""
    
    def __init__(self):
        self.index = None
        self.documents = None
        self.groq_client = None
        self.load_rag_components()
    
    def load_rag_components(self):
        """Load FAISS index and documents"""
        try:
            # Try to load FAISS components
            if Config.FAISS_INDEX_PATH and Config.FAISS_PKL_PATH:
                self.index = faiss.read_index(Config.FAISS_INDEX_PATH)
                with open(Config.FAISS_PKL_PATH, 'rb') as f:
                    self.documents = pickle.load(f)
            
            # Initialize Groq client
            self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
            st.success("✅ Enhanced RAG system with Llama loaded successfully")
        except Exception as e:
            st.warning(f"⚠️ RAG components partially loaded: {str(e)}")
            # Initialize Groq client even if FAISS fails
            try:
                self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
                st.success("✅ Llama model available via Groq API")
            except Exception as groq_e:
                st.error(f"❌ Groq API initialization failed: {str(groq_e)}")
    
    def search_relevant_docs(self, query_vector, k=3):
        """Search for relevant documents using FAISS"""
        if self.index is None:
            return ["No specific documentation available. Using general 5G security knowledge."]
        try:
            distances, indices = self.index.search(query_vector, k)
            if len(indices) > 0 and len(indices[0]) > 0:
                return [self.documents[i] for i in indices[0] if i < len(self.documents)]
            else:
                return ["No relevant documents found."]
        except Exception as e:
            st.error(f"Document search error: {str(e)}")
            return ["No specific documentation available. Using general 5G security knowledge."]
    
    def analyze_anomaly_type(self, anomaly_data):
        """Analyze the type of anomaly based on feature values"""
        anomaly_type = []
        severity = "Medium"
        
        # High latency check
        if anomaly_data.get('latency_ms', 0) > 50:
            anomaly_type.append("High Latency")
            severity = "High"
        
        # Resource exhaustion check
        if (anomaly_data.get('radio_resource_usage', 0) > 80 or 
            anomaly_data.get('core_network_load', 0) > 85 or 
            anomaly_data.get('upf_load', 0) > 90):
            anomaly_type.append("Resource Exhaustion")
            severity = "Critical"
        
        # Security threat indicators
        security_flags = [
            'slice_hopping_detected',
            'unauthorized_edge_access', 
            'abnormal_traffic_pattern',
            'qos_violation'
        ]
        
        active_security_flags = sum(1 for flag in security_flags if anomaly_data.get(flag, False))
        if active_security_flags >= 2:
            anomaly_type.append("Security Threat")
            severity = "Critical"
        elif active_security_flags == 1:
            anomaly_type.append("Potential Security Issue")
        
        # Performance degradation
        if anomaly_data.get('data_rate_mbps', 0) < 5:
            anomaly_type.append("Performance Degradation")
        
        return anomaly_type if anomaly_type else ["Unknown Anomaly"], severity
    
    def get_noc_engineer_solution(self, anomaly_data, relevant_docs=None):
        """Get comprehensive solution for NOC engineers using Llama via Groq"""
        if self.groq_client is None:
            return "❌ Llama model not available. Please check Groq API configuration."
        
        try:
            # Analyze anomaly type
            anomaly_types, severity = self.analyze_anomaly_type(anomaly_data)
            
            # Prepare context
            context = ""
            if relevant_docs:
                context = "\n".join(relevant_docs[:3])  # Use top 3 relevant docs
            
            # Create comprehensive prompt for Llama
            prompt = f"""
You are an expert 5G Network Operations Center (NOC) engineer AI assistant. Analyze the following anomaly data and provide actionable solutions.

ANOMALY ALERT DETAILS:
- Severity Level: {severity}
- Anomaly Types Detected: {', '.join(anomaly_types)}
- Device ID: {anomaly_data.get('device_id', 'Unknown')}
- Device Type: {anomaly_data.get('device_type', 'Unknown')}
- Network Slice: {anomaly_data.get('slice_type', 'Unknown')}

CRITICAL METRICS:
- Data Rate: {anomaly_data.get('data_rate_mbps', 'N/A')} Mbps
- Latency: {anomaly_data.get('latency_ms', 'N/A')} ms
- Radio Resource Usage: {anomaly_data.get('radio_resource_usage', 'N/A')}%
- Core Network Load: {anomaly_data.get('core_network_load', 'N/A')}%
- UPF Load: {anomaly_data.get('upf_load', 'N/A')}%

SECURITY INDICATORS:
- Slice Hopping: {'Yes' if anomaly_data.get('slice_hopping_detected') else 'No'}
- Unauthorized Edge Access: {'Yes' if anomaly_data.get('unauthorized_edge_access') else 'No'}
- Abnormal Traffic Pattern: {'Yes' if anomaly_data.get('abnormal_traffic_pattern') else 'No'}
- QoS Violation: {'Yes' if anomaly_data.get('qos_violation') else 'No'}

TECHNICAL CONTEXT:
{context}

Please provide a comprehensive NOC engineer response with the following sections:

1. IMMEDIATE ACTIONS (Next 5 minutes)
2. SHORT-TERM RESOLUTION (Next 30 minutes)
3. ROOT CAUSE ANALYSIS
4. PREVENTION STRATEGIES
5. ESCALATION CRITERIA
6. MONITORING RECOMMENDATIONS

Format your response clearly with specific technical commands, CLI commands where applicable, and step-by-step procedures. Focus on practical, actionable solutions that a NOC engineer can implement immediately.
            """
            
            # Call Llama via Groq API
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",  
                temperature=0.2,  # Lower temperature for more focused, technical responses
                max_tokens=2048
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Error generating NOC engineer solution: {str(e)}\n\nPlease check your Groq API configuration and try again."
    
    def get_quick_fix_commands(self, anomaly_data):
        """Generate quick CLI commands for immediate remediation"""
        commands = []
        
        # High resource usage commands
        if anomaly_data.get('core_network_load', 0) > 80:
            commands.extend([
                "# Check core network status",
                "kubectl get pods -n core-network",
                "# Scale up core network pods",
                "kubectl scale deployment core-network --replicas=5"
            ])
        
        if anomaly_data.get('upf_load', 0) > 85:
            commands.extend([
                "# Check UPF status and scale",
                "kubectl get pods -n upf",
                "kubectl scale deployment upf --replicas=3"
            ])
        
        # Security threat commands
        if anomaly_data.get('slice_hopping_detected'):
            commands.extend([
                "# Investigate slice hopping",
                "grep 'slice_hop' /var/log/5g-core/*.log",
                "# Block suspicious device temporarily",
                f"iptables -A INPUT -s {anomaly_data.get('device_id', 'DEVICE_IP')} -j DROP"
            ])
        
        if anomaly_data.get('unauthorized_edge_access'):
            commands.extend([
                "# Check edge access logs",
                "tail -f /var/log/edge-access.log | grep UNAUTHORIZED",
                "# Revoke edge certificates",
                "edge-cli revoke-cert --device-id " + anomaly_data.get('device_id', 'UNKNOWN')
            ])
        
        return commands

# Initialize components
@st.cache_resource
def init_components():
    """Initialize all system components"""
    db_manager = DatabaseManager()
    feature_processor = FeatureProcessor()
    rag_system = EnhancedRAGSystem()  # Using enhanced RAG system
    
    # Load ML model
    try:
        model = joblib.load(Config.MODEL_PATH)
        st.success("✅ ML model loaded successfully")
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        model = None
    
    return db_manager, feature_processor, rag_system, model

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ 5G NOC Anomaly Detection System</h1>', unsafe_allow_html=True)
    
    # Initialize components
    db_manager, feature_processor, rag_system, model = init_components()
    
    # Sidebar
    st.sidebar.title("🔧 System Controls")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Real-time Monitoring", "Request Analysis", "System Dashboard", "Historical Data"]
    )
    
    if page == "Real-time Monitoring":
        show_monitoring_page(db_manager, feature_processor, rag_system, model)
    elif page == "Request Analysis":
        show_analysis_page(db_manager, feature_processor, rag_system, model)
    elif page == "System Dashboard":
        show_dashboard_page(db_manager)
    elif page == "Historical Data":
        show_historical_page(db_manager)

def show_monitoring_page(db_manager, feature_processor, rag_system, model):
    """Real-time monitoring page with enhanced Llama integration"""
    
    st.header("📡 Real-time Network Monitoring")
    
    # Display important features info
    st.markdown("""
    <div class="important-features">
        <h4>🎯 Critical Features for Anomaly Detection:</h4>
        <p><strong>Data Rate (Mbps)</strong> | <strong>Latency (ms)</strong> | <strong>Radio Resource Usage (%)</strong> | <strong>Core Network Load (%)</strong> | <strong>UPF Load (%)</strong></p>
        <p><em>These features are mandatory for accurate anomaly detection.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input form for network request data
    with st.form("network_request_form"):
        st.subheader("📥 Network Request Input")
        
        # Basic Information
        st.markdown("### 📋 Basic Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            device_id = st.text_input("Device ID", value="DEV_001")
            device_type = st.selectbox("Device Type", 
                ['smartphone', 'iot_sensor', 'autonomous_vehicle', 'industrial_device'])
            slice_type = st.selectbox("Slice Type", ['eMBB', 'URLLC', 'mMTC'])
        
        with col2:
            active_sensors = st.number_input("Active Sensors", min_value=0, max_value=100, value=5)
            qos_class = st.selectbox("QoS Class", 
                ['best_effort', 'mission_critical', 'high_throughput'])
            edge_enabled = st.checkbox("Edge Enabled", value=True)
        
        with col3:
            auth_method = st.selectbox("Auth Method", 
                ['certificates', 'tokens', 'biometric'])
            encryption_level = st.selectbox("Encryption Level", 
                ['AES_128', 'AES_256', 'None'])
            network_isolation = st.selectbox("Network Isolation", 
                ['full', 'partial', 'none'])
        
        # Critical Features Section
        st.markdown("### ⚡ Critical Performance Metrics (Mandatory)")
        col4, col5 = st.columns(2)
        
        with col4:
            data_rate_mbps = st.number_input("⭐ Data Rate (Mbps)", min_value=0.0, value=10.5, 
                                           help="Critical feature for anomaly detection")
            latency_ms = st.number_input("⭐ Latency (ms)", min_value=0.0, value=15.2,
                                       help="Critical feature for anomaly detection")
            radio_resource_usage = st.slider("⭐ Radio Resource Usage (%)", 0, 100, 45,
                                            help="Critical feature for anomaly detection")
        
        with col5:
            core_network_load = st.slider("⭐ Core Network Load (%)", 0, 100, 65,
                                        help="Critical feature for anomaly detection")
            upf_load = st.slider("⭐ UPF Load (%)", 0, 100, 30,
                               help="Critical feature for anomaly detection")
        
        # Security Indicators
        st.markdown("### 🔒 Security Indicators")
        col6, col7 = st.columns(2)
        
        with col6:
            slice_hopping_detected = st.checkbox("Slice Hopping Detected")
            unauthorized_edge_access = st.checkbox("Unauthorized Edge Access")
        
        with col7:
            abnormal_traffic_pattern = st.checkbox("Abnormal Traffic Pattern")
            qos_violation = st.checkbox("QoS Violation")
        
        # Optional: Manual anomaly flag for training data
        st.markdown("### 🏷️ Ground Truth (Optional)")
        is_anomaly = st.checkbox("Mark as Anomaly (for training purposes)", value=False)
        
        submitted = st.form_submit_button("🔍 Analyze Request", type="primary")
        
        if submitted:
            # Prepare complete request data
            request_data = {
                'device_id': device_id,
                'device_type': device_type,
                'slice_type': slice_type,
                'active_sensors': active_sensors,
                'data_rate_mbps': data_rate_mbps,
                'latency_ms': latency_ms,
                'qos_class': qos_class,
                'edge_enabled': edge_enabled,
                'auth_method': auth_method,
                'encryption_level': encryption_level,
                'network_isolation': network_isolation,
                'radio_resource_usage': radio_resource_usage,
                'core_network_load': core_network_load,
                'upf_load': upf_load,
                'slice_hopping_detected': slice_hopping_detected,
                'unauthorized_edge_access': unauthorized_edge_access,
                'abnormal_traffic_pattern': abnormal_traffic_pattern,
                'qos_violation': qos_violation,
                'is_anomaly': is_anomaly
            }
            
            # Validate important features
            missing_features = feature_processor.validate_important_features(request_data)
            if missing_features:
                st.error(f"Missing critical features: {', '.join(missing_features)}")
                return
            
            # Store complete request data
            db_manager.insert_request_data(request_data, Config.ALL_REQUESTS_COLLECTION)
            
            # Extract features for prediction (only important ones)
            if model is not None:
                feature_vector = feature_processor.extract_important_features(request_data)
                
                # Display the features being used for prediction
                st.subheader("🎯 Features Used for Prediction")
                feature_df = pd.DataFrame({
                    'Feature': Config.IMPORTANT_FEATURES,
                    'Value': feature_vector[0]
                })
                st.dataframe(feature_df, use_container_width=True)
                
                # Make prediction
                prediction = model.predict(feature_vector)[0]
                prediction_proba = model.predict_proba(feature_vector)[0]
                
                # Display results
                st.subheader("🎯 Analysis Results")
                
                if prediction == 0:  # Normal request
                    st.markdown("""
                    <div class="normal-status">
                        <h3>✅ NORMAL REQUEST</h3>
                        <p>Access granted. Request patterns are within normal parameters.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Store in normal collection
                    result_data = request_data.copy()
                    result_data.update({
                        'prediction': int(prediction),
                        'normal_probability': float(prediction_proba[0]),
                        'anomaly_probability': float(prediction_proba[1])
                    })
                    db_manager.insert_request_data(result_data, Config.NORMAL_COLLECTION)
                    
                else:  # Anomaly detected - Enhanced Llama integration
                    st.markdown("""
                    <div class="anomaly-alert">
                        <h3>🚨 ANOMALY DETECTED</h3>
                        <p>Potential security threat identified. Immediate NOC engineer attention required.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Store in anomaly collection
                    result_data = request_data.copy()
                    result_data.update({
                        'prediction': int(prediction),
                        'normal_probability': float(prediction_proba[0]),
                        'anomaly_probability': float(prediction_proba[1])
                    })
                    db_manager.insert_request_data(result_data, Config.ANOMALY_COLLECTION)
                    
                    # Enhanced NOC Engineer Solutions using Llama
                    st.markdown("""
                    <div class="llama-solution">
                        <h2>🤖 Llama AI - NOC Engineer Solutions</h2>
                        <p><strong>Powered by Llama 3 70B via Groq API</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.spinner("🧠 Llama is analyzing the anomaly and generating NOC engineer solutions..."):
                        # Get relevant documents (if available)
                        query_vector = np.random.random((1, 768)).astype('float32')  # Placeholder
                        relevant_docs = rag_system.search_relevant_docs(query_vector)
                        
                        # Get comprehensive NOC engineer solution from Llama
                        noc_solution = rag_system.get_noc_engineer_solution(request_data, relevant_docs)
                        
                        # Display the Llama-generated solution
                        st.markdown("### 🛠️ Comprehensive NOC Engineer Action Plan")
                        st.markdown(f"""
                        <div class="solution-section">
                            {noc_solution.replace('\\n', '<br>')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Quick fix commands section
                        st.markdown("### ⚡ Quick Fix Commands")
                        quick_commands = rag_system.get_quick_fix_commands(request_data)
                        
                        if quick_commands:
                            st.code('\n'.join(quick_commands), language='bash')
                        else:
                            st.info("No immediate CLI commands available for this anomaly type.")
                        
                        # Additional analysis tabs
                        tab1, tab2, tab3 = st.tabs(["📊 Anomaly Analysis", "🔍 Historical Context", "📝 Report Summary"])
                        
                        with tab1:
                            anomaly_types, severity = rag_system.analyze_anomaly_type(request_data)
                            st.markdown(f"**Anomaly Types:** {', '.join(anomaly_types)}")
                            st.markdown(f"**Severity Level:** {severity}")
                            
                            # Create severity indicator
                            if severity == "Critical":
                                st.error(f"🔴 {severity} - Immediate escalation required")
                            elif severity == "High":
                                st.warning(f"🟡 {severity} - Urgent attention needed")
                            else:
                                st.info(f"🔵 {severity} - Monitor closely")
                        
                        with tab2:
                            # Show similar historical anomalies
                            similar_anomalies = db_manager.get_recent_requests(Config.ANOMALY_COLLECTION, 5)
                            if similar_anomalies:
                                st.markdown("**Recent Similar Anomalies:**")
                                for i, anom in enumerate(similar_anomalies[:3]):
                                    st.markdown(f"- Device: {anom.get('device_id', 'Unknown')} at {anom.get('timestamp', 'Unknown time')}")
                            else:
                                st.info("No recent similar anomalies found.")
                        
                        with tab3:
                            # Generate a summary report
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            report = f"""
**ANOMALY REPORT - {timestamp}**

**Device Information:**
- Device ID: {device_id}
- Device Type: {device_type}
- Network Slice: {slice_type}

**Anomaly Classification:**
- Types: {', '.join(anomaly_types)}
- Severity: {severity}
- Confidence: {prediction_proba[1]:.2%}

**Critical Metrics:**
- Data Rate: {data_rate_mbps} Mbps
- Latency: {latency_ms} ms
- Radio Resource Usage: {radio_resource_usage}%
- Core Network Load: {core_network_load}%
- UPF Load: {upf_load}%

**Security Flags:**
- Slice Hopping: {'Yes' if slice_hopping_detected else 'No'}
- Unauthorized Edge Access: {'Yes' if unauthorized_edge_access else 'No'}
- Abnormal Traffic: {'Yes' if abnormal_traffic_pattern else 'No'}
- QoS Violation: {'Yes' if qos_violation else 'No'}

**NOC Engineer Action Required:** YES
**Escalation Level:** {severity}
                            """
                            st.markdown(report)
                            
                            # Store report in session state for download outside form
                            st.session_state.anomaly_report = report
                            st.session_state.report_filename = f"anomaly_report_{device_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                            st.info("Report generated. Download button will appear below the form.")
          

                
                # Display prediction confidence
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Normal Probability", f"{prediction_proba[0]:.2%}")
                with col2:
                    st.metric("Anomaly Probability", f"{prediction_proba[1]:.2%}")
                
                # Feature importance visualization
                st.subheader("📊 Feature Analysis")
                fig = px.bar(
                    x=Config.IMPORTANT_FEATURES,
                    y=feature_vector[0],
                    title="Current Request Feature Values",
                    labels={'x': 'Features', 'y': 'Values'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("Model not loaded properly")
                
# Download button outside the form
if hasattr(st.session_state, 'anomaly_report'):
    st.download_button(
        label="📄 Download Report",
        data=st.session_state.anomaly_report,
        file_name=st.session_state.report_filename,
        mime="text/plain"
    )                 

def show_analysis_page(db_manager, feature_processor, rag_system, model):
    """Request analysis page"""
    
    st.header("📊 Request Analysis")
    
    # Recent requests analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Recent Normal Requests")
        normal_requests = db_manager.get_recent_requests(Config.NORMAL_COLLECTION, 5)
        for i, req in enumerate(normal_requests):
            with st.expander(f"Device: {req.get('device_id', 'Unknown')} - {req.get('timestamp', '')}"):
                # Display important features prominently
                st.markdown("**Critical Features:**")
                important_data = {k: req.get(k, 'N/A') for k in Config.IMPORTANT_FEATURES if k in req}
                st.json(important_data)
                
                st.markdown("**Complete Data:**")
                st.json({k: v for k, v in req.items() if k not in ['_id', 'timestamp']})
    
    with col2:
        st.subheader("🚨 Recent Anomalies")
        anomaly_requests = db_manager.get_recent_requests(Config.ANOMALY_COLLECTION, 5)
        for i, req in enumerate(anomaly_requests):
            with st.expander(f"⚠️ Device: {req.get('device_id', 'Unknown')} - {req.get('timestamp', '')}"):
                # Display important features prominently
                st.markdown("**Critical Features:**")
                important_data = {k: req.get(k, 'N/A') for k in Config.IMPORTANT_FEATURES if k in req}
                st.json(important_data)
                
                st.markdown("**Complete Data:**")
                st.json({k: v for k, v in req.items() if k not in ['_id', 'timestamp']})
                
                # Add button to regenerate Llama solution for this anomaly
                if st.button(f"🤖 Get Llama Solution", key=f"llama_btn_{i}"):
                    with st.spinner("Generating NOC solution with Llama..."):
                        solution = rag_system.get_noc_engineer_solution(req)
                        st.markdown("### 🛠️ Llama NOC Solution:")
                        st.markdown(solution)

def show_dashboard_page(db_manager):
    """System dashboard page"""
    
    st.header("📈 System Dashboard")
    
    # Get statistics
    normal_count = len(db_manager.get_recent_requests(Config.NORMAL_COLLECTION, 1000))
    anomaly_count = len(db_manager.get_recent_requests(Config.ANOMALY_COLLECTION, 1000))
    total_requests = normal_count + anomaly_count
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Requests", total_requests)
    with col2:
        st.metric("Normal Requests", normal_count)
    with col3:
        st.metric("Anomalies Detected", anomaly_count)
    with col4:
        anomaly_rate = (anomaly_count / total_requests * 100) if total_requests > 0 else 0
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
    
    # Feature Statistics
    st.subheader("📊 Critical Feature Statistics")
    
    # Get recent data for analysis
    recent_normal = db_manager.get_recent_requests(Config.NORMAL_COLLECTION, 100)
    recent_anomaly = db_manager.get_recent_requests(Config.ANOMALY_COLLECTION, 100)
    
    if recent_normal or recent_anomaly:
        # Create dataframes for analysis
        normal_df = pd.DataFrame(recent_normal)
        anomaly_df = pd.DataFrame(recent_anomaly)
        
        # Analyze important features
        for feature in Config.IMPORTANT_FEATURES:
            if feature in normal_df.columns or feature in anomaly_df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    if feature in normal_df.columns and len(normal_df) > 0:
                        normal_mean = normal_df[feature].mean()
                        st.metric(f"Normal {feature} (avg)", f"{normal_mean:.2f}")
                
                with col2:
                    if feature in anomaly_df.columns and len(anomaly_df) > 0:
                        anomaly_mean = anomaly_df[feature].mean()
                        st.metric(f"Anomaly {feature} (avg)", f"{anomaly_mean:.2f}")
    
    # Visualization
    if total_requests > 0:
        fig = px.pie(
            values=[normal_count, anomaly_count],
            names=['Normal', 'Anomaly'],
            title="Request Distribution",
            color_discrete_map={'Normal': 'green', 'Anomaly': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # NOC Engineer Dashboard
    st.subheader("🚨 NOC Engineer Alert Summary")
    
    if anomaly_count > 0:
        recent_anomalies = db_manager.get_recent_requests(Config.ANOMALY_COLLECTION, 10)
        
        # Create summary table
        anomaly_summary = []
        for anom in recent_anomalies:
            severity = "Critical" if (anom.get('core_network_load', 0) > 85 or 
                                   anom.get('upf_load', 0) > 90) else "Medium"
            anomaly_summary.append({
                'Device ID': anom.get('device_id', 'Unknown'),
                'Timestamp': anom.get('timestamp', 'Unknown'),
                'Severity': severity,
                'Data Rate': f"{anom.get('data_rate_mbps', 0):.1f} Mbps",
                'Latency': f"{anom.get('latency_ms', 0):.1f} ms",
                'Security Flags': sum([
                    anom.get('slice_hopping_detected', False),
                    anom.get('unauthorized_edge_access', False),
                    anom.get('abnormal_traffic_pattern', False),
                    anom.get('qos_violation', False)
                ])
            })
        
        if anomaly_summary:
            summary_df = pd.DataFrame(anomaly_summary)
            st.dataframe(summary_df, use_container_width=True)
            
            # Critical alerts count
            critical_count = sum(1 for item in anomaly_summary if item['Severity'] == 'Critical')
            if critical_count > 0:
                st.error(f"🔴 {critical_count} Critical alerts requiring immediate attention!")
    else:
        st.success("✅ No recent anomalies detected. System operating normally.")

def show_historical_page(db_manager):
    """Historical data page with enhanced analytics"""
    
    st.header("📚 Historical Data Analysis")
    
    # Get all historical data
    all_requests = db_manager.get_recent_requests(Config.ALL_REQUESTS_COLLECTION, 1000)
    
    if all_requests:
        df = pd.DataFrame(all_requests)
        
        st.subheader("📈 Critical Features Over Time")
        
        # Time series analysis of important features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            for feature in Config.IMPORTANT_FEATURES:
                if feature in df.columns:
                    fig = px.line(
                        df.sort_values('timestamp'),
                        x='timestamp',
                        y=feature,
                        title=f"{feature} Over Time",
                        color='is_anomaly' if 'is_anomaly' in df.columns else None
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Feature Correlation Analysis")
        
        # Correlation matrix for important features
        important_features_df = df[Config.IMPORTANT_FEATURES]
        if not important_features_df.empty:
            correlation_matrix = important_features_df.corr()
            
            fig = px.imshow(
                correlation_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly patterns analysis
        st.subheader("🔍 Anomaly Pattern Analysis")
        
        if 'is_anomaly' in df.columns:
            anomaly_df = df[df['is_anomaly'] == True]
            
            if not anomaly_df.empty:
                # Device type anomaly distribution
                device_anomalies = anomaly_df['device_type'].value_counts()
                fig = px.bar(
                    x=device_anomalies.index,
                    y=device_anomalies.values,
                    title="Anomalies by Device Type",
                    labels={'x': 'Device Type', 'y': 'Anomaly Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Network slice anomaly distribution
                if 'slice_type' in anomaly_df.columns:
                    slice_anomalies = anomaly_df['slice_type'].value_counts()
                    fig = px.pie(
                        values=slice_anomalies.values,
                        names=slice_anomalies.index,
                        title="Anomalies by Network Slice Type"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Export functionality
        st.subheader("📤 Data Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export All Data"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"5g_noc_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🚨 Export Anomalies Only"):
                if 'is_anomaly' in df.columns:
                    anomaly_data = df[df['is_anomaly'] == True]
                    csv = anomaly_data.to_csv(index=False)
                    st.download_button(
                        label="Download Anomalies CSV",
                        data=csv,
                        file_name=f"5g_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col3:
            if st.button("📊 Export Feature Stats"):
                stats = df[Config.IMPORTANT_FEATURES].describe()
                csv = stats.to_csv()
                st.download_button(
                    label="Download Stats CSV",
                    data=csv,
                    file_name=f"5g_feature_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
    else:
        st.info("No historical data available. Start monitoring to collect data.")

if __name__ == "__main__":
    main()