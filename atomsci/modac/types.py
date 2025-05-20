from typing import TypedDict, List, Dict


class DataObject(TypedDict):
    dataSize: int
    id: int
    path: str


class SubCollection(TypedDict):
    dataSize: int
    id: int
    path: str


class Collection(TypedDict):
    absolutePath: str
    collectionId: int
    collectionInheritance: str
    collectionMapId: str
    collectionName: str
    collectionOwnerName: str
    collectionOwnerZone: str
    collectionParentName: str
    createdAt: int
    dataObjects: List[DataObject]
    dataObjectsTotalRecords: int
    specColType: str
    subCollections: List[SubCollection]
    subCollectionsTotalRecords: int


class MetadataEntry(TypedDict):
    attribute: str
    collectionId: int
    level: int
    levelLabel: str
    unit: str
    value: str


class CollectionResponse(TypedDict):
    collection: Collection
    metadataEntries: Dict[str, List[MetadataEntry]]
