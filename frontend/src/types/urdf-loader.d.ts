declare module "urdf-loader" {
  import type { Object3D, LoadingManager } from "three";

  export interface URDFJointLimit {
    lower?: number;
    upper?: number;
  }

  export interface URDFJoint extends Object3D {
    jointType: string;
    limit?: URDFJointLimit;
    angle?: number;
    setJointValue(value: number): void;
  }

  export interface URDFLink extends Object3D {
    isURDFLink: boolean;
  }

  export interface URDFRobot extends Object3D {
    joints: Record<string, URDFJoint>;
    links: Record<string, URDFLink>;
  }

  export default class URDFLoader {
    constructor(manager?: LoadingManager);
    workingPath: string;
    loadMeshCb: (
      path: string,
      manager: LoadingManager,
      onComplete: (obj: Object3D | null, err?: Error) => void,
    ) => void;
    parse(content: string): URDFRobot;
  }
}
