from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import uuid


class PatientName(BaseModel):
    firstName: Optional[str] = None
    middleName: Optional[str] = None
    lastName: Optional[str] = None


class GSSampleID(BaseModel):
    sampleA: Optional[str] = None
    sampleB: Optional[str] = None


class PatientInformation(BaseModel):
    mrnUhid: Optional[str] = None
    patientName: Optional[PatientName] = None
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    comorbidities: Optional[str] = None
    dob: Optional[str] = None
    age: Optional[str] = None
    weight: Optional[str] = None
    height: Optional[str] = None
    email: Optional[str] = None
    patientInformationPhoneNumber: Optional[str] = None
    patientInformationAddress: Optional[str] = None
    patientCountry: Optional[str] = None
    patientState: Optional[str] = None
    patientCity: Optional[str] = None
    patientPincode: Optional[str] = None
    patientStreetAddress: Optional[str] = None
    patientOncologist: Optional[str] = None
    patientHospital: Optional[str] = None


class Approver(BaseModel):
    emailId: Optional[str] = None
    name: Optional[str] = None
    _id: Optional[str] = None
    signed: bool = False
    signedDate: Optional[str] = None


class ApproversData(BaseModel):
    userEmail: Optional[str] = None
    approvers: List[Approver] = []
    oncologists: List[Approver] = []
    finalApproverEmailId: Optional[str] = None
    finalApproverName: Optional[str] = None
    userSigned: bool = False
    userSignedDate: Optional[str] = None
    userName: Optional[str] = None
    finalSigned: bool = False
    finalSignedDate: Optional[str] = None
    moRequestedSignature: bool = False
    patientSigned: bool = False


class GenesilicoKarkinos(BaseModel):
    accessioning: bool = False
    accessioningDate: Optional[str] = None
    dnaqc: bool = False
    dnaqcDate: Optional[str] = None
    ngsdataqc: bool = False
    ngsdataqcDate: Optional[str] = None
    ngslibraryqc: bool = False
    ngslibraryqcDate: Optional[str] = None
    sequencing: bool = False
    sequencingDate: Optional[str] = None
    ngsreport: bool = False
    ngsreportDate: Optional[str] = None
    payment: Optional[bool] = None
    plan: Optional[str] = None
    subscriptionStart: Optional[str] = None


class immunohistochemistry(BaseModel):
    er: Optional[str] = None
    pr: Optional[str] = None
    her2neu: Optional[str] = None
    ki67: Optional[str] = None
    diseaseStatusAtTheTimeOfTesting: List[str] = []
    hasPatientFailedPriorTreatment: Optional[str] = None
    pastTherapy: List[str] = []
    currentTherapy: List[str] = []


class ClinicalSummary(BaseModel):
    primaryDiagnosis: Optional[str] = None
    initialDiagnosisStage: Optional[str] = None
    currentDiagnosis: Optional[str] = None
    diagnosisDate: Optional[str] = None
    genesilicoStudy: Optional[str] = None
    ageOfManifestation: Optional[str] = None
    Immunohistochemistry: Optional[immunohistochemistry] = None
    pastTherapyComment: Optional[str] = None
    currentTherapyComment: Optional[str] = None
    pastIllness: Optional[str] = None
    comments: Optional[str] = None


class FamilyMember(BaseModel):
    relationToPatient: Optional[str] = None
    typesOfCancer: Optional[str] = None
    EstimatedAgeAtOnset: Optional[str] = None


class familyHistory(BaseModel):
    familyHistoryOfAnyCancer: Optional[str] = None
    familyMember: List[FamilyMember] = []
    comments: Optional[str] = None


class Hospital(BaseModel):
    hospitalID: Optional[str] = None
    hospitalName: Optional[str] = None
    hospitalAddress: Optional[str] = None
    country: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    postalCode: Optional[str] = None
    contactPersonNameHospital: Optional[str] = None
    contactPersonPhoneHospital: Optional[str] = None
    contactPersonEmailHospital: Optional[str] = None
    patientStatusAtTheTimeOfSampleCollection: Optional[str] = None


class Physician(BaseModel):
    physicianOncologist: Optional[str] = None
    physicianName: Optional[str] = None
    physicianSpecialty: Optional[str] = None
    physicianPhoneNumber: Optional[str] = None
    physicianEmail: Optional[str] = None


class lab(BaseModel):
    labID: Optional[str] = None
    labName: Optional[str] = None
    labAddress: Optional[str] = None
    submittingPathologistLabInchargeName: Optional[str] = None
    labPhoneNumber: Optional[str] = None
    email: Optional[str] = None
    fax: Optional[str] = None
    country: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    postalCode: Optional[str] = None


class Sample(BaseModel):
    sampleCollectionType: Optional[str] = None
    sampleType: Optional[str] = None
    sampleID: Optional[str] = None
    sampleCollectionSite: List[str] = []
    sampleCollectionDate: Optional[str] = None
    timeOfCollection: Optional[str] = None
    currentStatusOfSample: List[str] = []
    selectTheTemperatureAtWhichItIsStored: Optional[str] = None
    sampleUHID: Optional[str] = None
    storedIn: List[str] = []


class ShipmentDetails(BaseModel):
    sampleDestination: Optional[str] = None
    typeOfLogistics: Optional[str] = None
    pickupLocation: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    address: Optional[str] = None
    postalCode: Optional[str] = None
    contactPersonName: Optional[str] = None
    contactNumber: Optional[str] = None
    emailID: Optional[str] = None
    pickupDate: Optional[str] = None
    pickupTime: Optional[str] = None
    agentName: Optional[str] = None
    agentContactNumber: Optional[str] = None
    courierTrackingNumber: Optional[str] = None
    shipmentFlag: Optional[bool] = None
    shipmentSave: Optional[bool] = None


class Consent(BaseModel):
    dateOfConsent: Optional[str] = None
    patientConsent: Optional[bool] = None
    patientSampleConsent: Optional[bool] = None
    gcConsent: Optional[bool] = None
    patientOrGuardianSignature: Optional[str] = None
    guardianRelationship: Optional[str] = None


class Reviewers(BaseModel):
    reviewer1Name: Optional[str] = None
    reviewer1Date: Optional[str] = None
    reviewer2Name: Optional[str] = None
    reviewer2Date: Optional[str] = None
    reviewer3Name: Optional[str] = None
    reviewer3Date: Optional[str] = None


class Prospective(BaseModel):
    digitalSignature: Optional[str] = None
    signatureDate: Optional[str] = None


class FinalApprovalAndSignature(BaseModel):
    digitalSignature: Optional[str] = None
    printedFullName: Optional[str] = None
    finalApprovalAndSignatureDate: Optional[str] = None


class PerformanceStatus(BaseModel):
    geriatricScore: Optional[str] = None
    ecogStatus: Optional[str] = None
    karnofskyScore: Optional[str] = None
    lastUpdatedDate: datetime = Field(default_factory=datetime.now)


class Staging(BaseModel):
    tStage: Optional[str] = None
    nStage: Optional[str] = None
    mStage: Optional[str] = None
    behavior: Optional[str] = None
    differentiation: Optional[str] = None
    otherInfo: Optional[str] = None


class SurgicalDetails(BaseModel):
    surgeryDate: Optional[str] = None
    intentTiming: Optional[str] = None
    surgeryName: Optional[str] = None
    pathologicalTumorSize: Optional[str] = None


class PatientReport(BaseModel):
    """TRF data model representing a patient report (test requisition form)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    viewOnlyMode: bool = False
    reportId: Optional[str] = None
    patientID: str
    limsID: Optional[str] = None
    gssampleID: List[GSSampleID] = []
    orderID: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None
    patientActivationStatus: bool = True
    email: Optional[str] = None
    formStatus: Optional[str] = None
    initiatedDate: datetime = Field(default_factory=datetime.now)
    initiator: Optional[str] = None
    lastUpdated: datetime = Field(default_factory=datetime.now)
    patientInformation: Optional[PatientInformation] = None
    approversData: Optional[ApproversData] = None
    genesilicoKarkinos: Optional[GenesilicoKarkinos] = None
    clinicalSummary: Optional[ClinicalSummary] = None
    FamilyHistory: Optional[familyHistory] = None
    hospital: Optional[Hospital] = None
    physician: Optional[Physician] = None
    Records: List = []
    Lab: Optional[lab] = None
    Sample: Dict[str, Any] = {}  # Use a plain List without type parameter for compatibility
    shipmentDetails: Optional[ShipmentDetails] = None
    consent: Optional[Consent] = None
    reviewers: Optional[Reviewers] = None
    prospective: Optional[Prospective] = None
    finalApprovalandSignature: Optional[FinalApprovalAndSignature] = None
    medicalCoordinatorAcknowledgement: Optional[str] = None
    medicalCoordinatorDate: Optional[str] = None
    medicalCoordinatorMailID: Optional[str] = None
    progressBar: Optional[int] = None
    performanceStatus: Optional[PerformanceStatus] = None
    staging: Optional[Staging] = None
    surgicalDetails: Optional[SurgicalDetails] = None
    qaigenRecord: List = []
    
    # Metadata
    document_id: Optional[str] = None
    ocr_result_id: Optional[str] = None
    extraction_confidence: Optional[float] = None
    extraction_time: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Extracted fields tracking
    extracted_fields: Dict[str, float] = {}
    missing_required_fields: List[str] = []
    low_confidence_fields: List[str] = []
    
    class Config:
        populate_by_name = True
